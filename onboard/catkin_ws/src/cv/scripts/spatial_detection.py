#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np


class DepthAISpatialDetector:
    def __init__(self, nnBlobPath=str((Path(__file__).parent / Path('tiny-yolo-v4_openvino_2021.2_6shave.blob')).resolve().absolute())):

        if not Path(nnBlobPath).exists():
            import sys
            raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

        self.pipeline = self.get_pipeline(nnBlobPath)
        self.output_queues = {}

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
        spatialDetectionNetwork.setNumClasses(80)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
        spatialDetectionNetwork.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
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

    def connect_and_get_output_queues(self):
        with dai.Device(self.pipeline) as device:
            self.output_queues["rgb"] = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            self.output_queues["detections"] = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
            self.output_queues["boundingBoxDepthMapping"] = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=1, blocking=False)
            self.output_queues["depth"] = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

    def run_detection(self):
        inPreview = self.output_queues["rgb"].get()
        inDet = self.output_queues["detections"].get()
        depth = self.output_queues["depth"].get()

        frame = inPreview.getCvFrame()
        detections = inDet.detections

        height = frame.shape[0]
        width = frame.shape[1]
        for detection in detections:

            bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)

            label = detection.label

            confidence = detection.confidence
            x = detection.spatialCoordinates.x
            y = detection.spatialCoordinates.y
            z = detection.spatialCoordinates.z

            print(f'Label: {label}, Confidence: {confidence}, X: {x}, Y: {y}, Z: {z}')


if __name__ == '__main__':
    depthai_detector = DepthAIDetector()
    depthai_detector.run_detection()
