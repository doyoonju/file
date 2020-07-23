#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import os, rospkg #os
import json

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridgeError
from std_msgs.msg import Float64
from utils import PID_longitudinal, purePursuit

if __name__ == "__main__" :

    rospy.init_node("traffic_control", anonymous=True)
    ctrl_lon = PID_longitudinal(K=500)
    ctrl_lat = purePursuit(lfd=0.8)
    rate = rospy.Rate(20)

    while not rospy.is_shutdown():

        if ctrl_lon.sline is not None and ctrl_lat.lpath is not None:
            ctrl_lon.calc_val_cmd()
            ctrl_lon.pub_cmd()
            ctrl_lat.steering_angle()
            ctrl_lat.pub_cmd()
            rate.sleep()