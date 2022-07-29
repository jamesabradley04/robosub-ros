#!/usr/bin/env python3

import rospy
import resource_retriever as rr
import yaml

from pathlib import Path
import cv2
import depthai as dai
import numpy as np

from custom_msgs.srv import EnableModel
from custom_msgs.msg import CVObject
from sensor_msgs.msg import Image


class DepthAISpatialDetector:
    def __init__(self):
        rospy.init_node('cv', anonymous=True)

        with open(rr.get_filename('package://cv/models/spatial_detection_models.yaml',
                                  use_protocol=False)) as f:
            self.models = yaml.safe_load(f)

        self.camera = 'front'
        self.pipeline = None
        self.publishers = None
        self.output_queues = {}
        self.connected = False
        self.current_model_name = None
        self.classes = None

        self.enable_service = f'enable_model_{self.camera}'

    def get_pipeline(self, nnBlobPath, syncNN=True):
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutBoundingBoxDepthMapping = pipeline.create(dai.node.XLinkOut)
        xoutDepth = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        xoutNN.setStreamName("detections")
        xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
        xoutDepth.setStreamName("depth")

        # Properties
        camRgb.setPreviewSize(416, 416)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

        spatialDetectionNetwork.setBlobPath(nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(5)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
        spatialDetectionNetwork.setAnchorMasks({ "side26": [0, 1, 2], "side13": [3,4,5] })
        spatialDetectionNetwork.setIouThreshold(0.5)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)
        if syncNN:
            spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        else:
            camRgb.preview.link(xoutRgb.input)

        spatialDetectionNetwork.out.link(xoutNN.input)
        spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

        return pipeline

    def init_model(self, model_name):
        if model_name == self.current_model_name:
            return
        self.current_model_name = model_name

        model = self.models[model_name]

        self.classes = model['classes']

        blob_path = rr.get_filename(f"package://cv/models/{model['weights']}",
                                    use_protocol=False)
        self.pipeline = self.get_pipeline(blob_path)

        # For now, creating publisher for each class
        publisher_dict = {}
        for model_class in model['classes']:
            publisher_name = f"cv/{self.camera}/{model_class}"
            publisher_dict[model_class] = rospy.Publisher(publisher_name,
                                                          CVObject,
                                                          queue_size=10)
        self.publishers = publisher_dict

    def get_output_queues(self, device):
        if self.connected:
            return

        self.output_queues["rgb"] = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.output_queues["detections"] = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
        self.output_queues["boundingBoxDepthMapping"] = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=1, blocking=False)
        self.output_queues["depth"] = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        self.connected = True

    def detect(self):
        if not self.connected:
            return

        inPreview = self.output_queues["rgb"].get()
        inDet = self.output_queues["detections"].get()
        depth = self.output_queues["depth"].get()

        frame = inPreview.getCvFrame()
        detections = inDet.detections

        height = frame.shape[0]
        width = frame.shape[1]
        for detection in detections:

            bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)

            label_idx = detection.label
            label = self.classes[label_idx]

            confidence = detection.confidence
            x = detection.spatialCoordinates.x
            y = detection.spatialCoordinates.y
            z = detection.spatialCoordinates.z

            # print(f'Label: {label}, Confidence: {confidence}, X: {x}, Y: {y}, Z: {z}')
            self.publish_prediction(bbox, z, label, confidence, (height, width))

    def publish_prediction(self, bbox, depth, label, confidence, shape):
        object_msg = CVObject()
        object_msg.label = label
        object_msg.score = confidence
        object_msg.distance = depth

        object_msg.xmin = bbox[0]
        object_msg.ymin = bbox[1]
        object_msg.xmax = bbox[2]
        object_msg.ymax = bbox[3]

        object_msg.height = shape[0]
        object_msg.width = shape[1]

        if self.publishers:
            self.publishers[label].publish(object_msg)

    def run_model(self, req):
        if not req.model_name in self.models:
            return False

        self.init_model(req.model_name)
        with dai.Device(self.pipeline) as device:
            self.get_output_queues(device)

            loop_rate = rospy.Rate(1)
            while not rospy.is_shutdown():
                self.detect()
                loop_rate.sleep()

        return True

    def run(self):
        rospy.Service(self.enable_service, EnableModel, self.run_model)
        rospy.spin()


if __name__ == '__main__':
    DepthAISpatialDetector().run()
