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


MM_IN_METER = 1000
SPATIAL_DETECTION_MODELS_FILEPATH = 'package://cv/models/spatial_detection_models.yaml'

class DepthAISpatialDetector:
    def __init__(self):
        """
        Initializes the ROS node and service. Loads the yaml file at cv/models/spatial_detection_models.yaml
        """
        rospy.init_node('cv', anonymous=True)

        with open(rr.get_filename(SPATIAL_DETECTION_MODELS_FILEPATH,
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
        """
        Get the DepthAI Pipeline for 3D object localization. Inspiration taken from
        https://docs.luxonis.com/projects/api/en/latest/samples/SpatialDetection/spatial_tiny_yolo/.
        To understand the DepthAI pipeline structure, please see https://docs.luxonis.com/projects/api/en/latest/.
        This pipeline computes the depth map using the two mono cameras. This depth map and the RGB feed are fed into
        the YoloSpatialDetection Node, which detects objects and computes the average depth within the bounding box
        for any detected object. The object detection model for this node is loaded from the nnBlobPath. For info
        about the YoloSpatialDetection Node, see
        https://docs.luxonis.com/projects/api/en/latest/components/nodes/yolo_spatial_detection_network/.
        The output queues available from this pipeline are:
            - "rgb": contains the 400x400 RGB preview of the camera feed.
            - "detections": contains SpatialImgDetections messages (https://docs.luxonis.com/projects/api/en/latest/components/messages/spatial_img_detections/#spatialimgdetections),
                            which includes bounding boxes for detections as well as XYZ coordinates of the detected objects.
            - "boundingBoxDepthMapping": contains SpatialLocationCalculatorConfig messages, which provide a mapping
                                         between the RGB feed from which bounding boxes are computed and the depth map.
            - "depth": contains ImgFrame messages with UINT16 values which represent the depth in millimeters by default.
                       see the depth output of https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/

        :param nnBlobPath: Path to blob file used for object detection.
        :param syncNN: If True, sync the RGB output feed with the detection from the neural network. Needed if the RGB
        feed output will be used and needs to be synced with the object detections.
        :return: depthai.Pipeline object to compute
        """
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
        """
        Creates and assigns the pipeline, sets the current model name, and creates the publishers.
        A publisher is created for each class that the model predicts. The publishers are created in format: "cv/camera/model_name".
        
        :param model_name: Name of the model. The model name should match a key in cv/models/spatial_detection_models.yaml.
        For example, if spatial_detection_models.yaml is

        gate:
            classes: [start_gate, start_tick]
            topic: /cv
            weights: gate-tick-15-epochs.pth

        Then model name is "gate".
        """
        if model_name == self.current_model_name:
            return
        self.current_model_name = model_name

        model = self.models[model_name]

        self.classes = model['classes']

        blob_path = rr.get_filename(f"package://cv/models/{model['weights']}",
                                    use_protocol=False)
        self.pipeline = self.get_pipeline(blob_path)

        publisher_dict = {}
        for model_class in model['classes']:
            publisher_name = f"cv/{self.camera}/{model_class}"
            publisher_dict[model_class] = rospy.Publisher(publisher_name,
                                                          CVObject,
                                                          queue_size=10)
        self.publishers = publisher_dict

    def get_output_queues(self, device):
        """
        Assigns output queues from the pipeline to dictionary of queues.
        :param device: DepthAI.Device object for the connected device. See https://docs.luxonis.com/projects/api/en/latest/components/device/
        """
        if self.connected:
            return

        self.output_queues["rgb"] = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.output_queues["detections"] = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
        self.output_queues["boundingBoxDepthMapping"] = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=1, blocking=False)
        self.output_queues["depth"] = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        self.connected = True

    def detect(self):
        """
        Get current detections from output queues and publish.
        """
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
            x_cam_mm = detection.spatialCoordinates.x  # x is left/right axis, where 0 is in middle of the frame, to the left is negative x, and to the right is positive x
            y_cam_mm = detection.spatialCoordinates.y  # y is down/up axis, where 0 is in the middle of the frame, down is negative y, and up is positive y
            z_cam_mm = detection.spatialCoordinates.z  # z is distance of object from camera in mm

            x_cam_meters, y_cam_meters, z_cam_meters = self.mm_to_meters(x_cam_mm), self.mm_to_meters(y_cam_mm), self.mm_to_meters(z_cam_mm)

            det_coords_robot_mm = self.camera_frame_to_robot_frame(x_cam_meters, y_cam_meters, z_cam_meters)

            self.publish_prediction(bbox, det_coords_robot_mm, label, confidence, (height, width))

    def publish_prediction(self, bbox, det_coords, label, confidence, shape):
        """
        Publish predictions to label-specific topic. Publishes to /cv/[camera]/[label].

        :param bbox: Tuple for the bounding box. Values are from 0-1, where X increases left to right and Y increases.
        :param det_coords: Tuple with the X, Y, and Z values in meters, and in the robot rotational reference frame.
        :param label: Predicted label for the detection.
        :param confidence: Confidence for the detection, from 0 to 1.
        :param shape: Tuple with the (height, width) of the image.
        """
        object_msg = CVObject()
        object_msg.label = label
        object_msg.score = confidence

        object_msg.x = det_coords[0]
        object_msg.y = det_coords[1]
        object_msg.z = det_coords[2]

        object_msg.xmin = bbox[0]
        object_msg.ymin = bbox[1]
        object_msg.xmax = bbox[2]
        object_msg.ymax = bbox[3]

        object_msg.height = shape[0]
        object_msg.width = shape[1]

        if self.publishers:
            self.publishers[label].publish(object_msg)

    def camera_frame_to_robot_frame(self, cam_x, cam_y, cam_z):
        """
        Convert coordinates in camera reference frame to coordinates in robot reference frame.
        This ONLY ACCOUNTS FOR THE ROTATION BETWEEN COORDINATE FRAMES, and DOES NOT ACCOUNT FOR THE TRANSLATION.
        :param cam_x: X coordinate of object in camera reference frame.
        :param cam_y: Y coordinate of object in camera reference frame.
        :param cam_z: Z coordinate of object in camera reference frame.
        :return: X,Y,Z coordinates of object in robot rotational reference frame.
        """
        robot_y = -cam_x
        robot_z = cam_y
        robot_x = cam_z
        return robot_x, robot_y, robot_z

    def mm_to_meters(self, val_mm):
        """
        Converts value from millimeters to meters.
        :param val_mm: Value in millimeters.
        :return: Input value converted to meters.
        """
        return val_mm / MM_IN_METER

    def run_model(self, req):
        """
        Runs the model on the connected device.
        :param req: Request from
        :return: False if the model is not in cv/models/spatial_detection_models.yaml. Otherwise, the model will be run on the device.
        """
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
        """
        Wait for the EnableModel rosservice call. If this call is made, the model specified will be run.
        For information about EnableModel, see robosub-ros/core/catkin_ws/src/custom_msgs/srv/EnableModel.srv

        Example ros commands to run this node and activate the model:
            roslaunch cv spatial_detection_front.launch
            rosservice call /enable_model_front gate True

        """
        rospy.Service(self.enable_service, EnableModel, self.run_model)
        rospy.spin()


if __name__ == '__main__':
    DepthAISpatialDetector().run()
