#!/usr/bin/env python3

import rospy
import yaml
from custom_msgs.msg import CVObject
from custom_msgs.srv import EnableModel
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from detecto.core import Model
import resource_retriever as rr


class Detector:

    # Load in models and other misc. setup work
    def __init__(self):
        rospy.init_node('cv', anonymous=True)

        # Input to CV pipeline from camera stream
        self.bridge = CvBridge()
        self.camera = rospy.get_param('~camera')

        # Load in model configurations as a dictionary
        with open(rr.get_filename('package://cv/models/models.yaml', use_protocol=False)) as f:
            self.model_outline = yaml.safe_load(f)

        # The topic that the camera publishes its feed to
        self.camera_feed_topic = f'/camera/{self.camera}/image_raw'

        # Toggle model service name
        self.enable_service = f'enable_model_{self.camera}'
        
        # LOAD MODEL AND CREATE PUBLISHER

    # Initialize model predictor if not already initialized
    def init_model(self, model_name):

        # GET MODEL_NAME PARAMETER (think about where we read in that info before)

        # Model already initialized; return from method
        if self.predictor is not None:
            return

        # DECLARE MODEL HERE

    # Camera subscriber callback; publishes predictions for each frame
    def detect(self, img_msg):

        # Read the current frame from the camera stream
        image = self.bridge.imgmsg_to_cv2(img_msg, 'rgb8')

        # CALL PREDICTOR TO MAKE A PREDICTION ON IMAGE

    # Publish predictions with the given publisher
    def publish_predictions(self, preds, publisher, shape):
        labels, boxes, scores = preds

        # If there are no predictions, publish 'none' as the object label
        if not labels:

            object_msg = CVObject()
            object_msg.label = 'none'
            
            # PUBLISH THE BLANK OBJECT_MSG

        else:
            for label, box, score in zip(labels, boxes, scores):
                object_msg = CVObject()

                object_msg.label = label
                object_msg.score = score

                object_msg.xmin = box[0].item() / shape[1]
                object_msg.ymin = box[1].item() / shape[0]
                object_msg.xmax = box[2].item() / shape[1]
                object_msg.ymax = box[3].item() / shape[0]

                object_msg.height = shape[0]
                object_msg.width = shape[1]

                # FIND PUBLISHER; IF FOUND, PUBLISH

    # Service for toggling specific models on and off
    def enable_model(self, req):

        # Retrieve model from model_outline yaml dict
        model = self.model_outline[self.model_name]
        model['enabled'] = req.enabled

        
        # WRITE CODE TO SEE IF MODEL IS ENABLED
        # IF NO, SET PREDICTOR AND PUBLISHER TO NONE
        # IF YES, INITIALIZE MODEL HERE

        return True

    # Initialize node and set up Subscriber to generate and
    # publish predictions at every camera frame
    def run(self):
        rospy.Subscriber(self.camera_feed_topic, Image, self.detect)

        # Allow service for toggling of models
        rospy.Service(self.enable_service, EnableModel, self.enable_model)

        # Keep node running until shut down
        rospy.spin()


if __name__ == '__main__':
    Detector().run()