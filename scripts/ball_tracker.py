#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from copy import deepcopy
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist, Vector3


class BallTracker(object):
    """ The BallTracker is a Python object that encompasses a ROS node 
        that can process images from the camera and search for a ball within.
        The node will issue motor commands to move forward while keeping
        the ball in the center of the camera's field of view. """

    def __init__(self, image_topic):
        """ Initialize the ball tracker """
        rospy.init_node('ball_tracker')
        self.cv_image = None  # the latest image from the camera
        self.binary_image = None
        self.bridge = CvBridge()  # used to convert ROS messages to OpenCV

        rospy.Subscriber(image_topic, Image, self.process_image)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        cv2.namedWindow('raw_video')
        cv2.namedWindow('video_window')
        cv2.namedWindow('threshold_image')

        self.xMean = 0
        self.yMean = 0
        self.xPixels = 0
        self.yPixels = 0

        self.minPixelsNeeded = 10
        self.turnP = rospy.get_param('~kTurn', 2.0)
        self.driveSpeed = rospy.get_param('~driveSpeed', 0.15)

        rospy.on_shutdown(lambda : self.pub.publish(Twist()))

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.hsv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        self.binary_image = cv2.inRange(self.hsv_image, (40, 120, 100), (65, 256, 256))

        moments = cv2.moments(self.binary_image, binaryImage=True)
        if moments['m00'] >= self.minPixelsNeeded:
            # I have sufficient information to process the image

            shape = self.cv_image.shape
            self.xPixels = moments['m10'] / moments['m00']
            self.xMean = (self.xPixels / shape[1]) - 0.5
            self.yPixels = moments['m01'] / moments['m00']
            self.yMean = -(self.yPixels / shape[0]) - 0.5

            # Drive the motors
            turnSpeed = -self.turnP * self.xMean
            forwardSpeed = self.driveSpeed
            self.pub.publish(Twist(linear=Vector3(x=forwardSpeed), angular=Vector3(z=turnSpeed)))

    def run(self):
        """ The main run loop, in this node it doesn't do anything """
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.cv_image is not None:
                print self.cv_image.shape
                cv2.circle(img=self.cv_image, center=(int(self.xPixels), int(self.yPixels)), radius=10, color=(255, 0, 0))
                cv2.imshow('raw_video', self.cv_image)
                cv2.imshow('video_window', self.hsv_image)
                cv2.imshow('threshold_image', self.binary_image)
                cv2.waitKey(5)

                print self.xMean, self.yMean


            r.sleep()


if __name__ == '__main__':
    node = BallTracker("/camera/image_raw")
    node.run()
