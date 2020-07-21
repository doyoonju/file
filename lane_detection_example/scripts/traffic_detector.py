#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import os, rospkg #os

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridgeError


class TRAFFICDetector:

    def __init__(self):

        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)

        self.traffic_msg = String()

        self.signal_pub = rospy.Publisher("/traffic_light", String, queue_size=10)

        self.image_hsv = None

    def callback (self, msg):
        try:
            np_arr = np.fromstring(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)

        self.image_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    def detect_signal(self):

        h = self.image_hsv.shape[0]

        lower_sig_r = np.array([0, 250, 250])
        upper_sig_r = np.array([15, 255, 255])

        lower_sig_y = np.array([20, 250, 250])
        upper_sig_y = np.array([35, 255, 255])

        lower_sig_g = np.array([50, 250, 250])
        upper_sig_g = np.array([70, 255, 255])

        img_r = cv2.inRange(self.image_hsv, lower_sig_r, upper_sig_r)
        img_y = cv2.inRange(self.image_hsv, lower_sig_y, upper_sig_y)
        img_g = cv2.inRange(self.image_hsv, lower_sig_g, upper_sig_g)

        img_r[int(h/3):,:] = 0
        img_y[int(h/3):,:] = 0
        img_g[int(h/3):,:] = 0

        pix_r = cv2.countNonZero(img_r)
        pix_y = cv2.countNonZero(img_y)
        pix_g = cv2.countNonZero(img_g)

        pix_max = np.max([pix_r, pix_y, pix_g])
        idx_s = np.argmax([pix_r, pix_y, pix_g])

        if pix_max>40:

            if idx_s ==0:
                self.traffic_msg.data = "RED"

            elif idx_s ==1:
                self.traffic_msg.data = "YELLOW"
            
            else:
                self.traffic_msg.data = "GREEN"

        else:
            self.traffic_msg.data = "None"

    def pub_signal(self):

        self.signal_pub.publish(self.traffic_msg)

if __name__ == "__main__" :

    rospy.init_node("traffic_detector", anonymous=True)

    traffic_detector = TRAFFICDetector()

    rate = rospy.Rate(20)

    while not rospy.is_shutdown():

        if traffic_detector.image_hsv is not None:

            traffic_detector.detect_signal()

            traffic_detector.pub_signal()

            rate.sleep()