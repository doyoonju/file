#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import os, rospkg #os
import json

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridgeError
# from std_msgs.msg import Float64
from utils import BEVTransform, CURVEFit, draw_lane_img#, warp_image, purePursuit,STOPLineEstimator


class IMGParser:
    def __init__(self):

        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)

        self.source_prop = np.float32([[0.05, 0.65],
                                       [0.5 - 0.15, 0.55],
                                       [0.5 + 0.15, 0.55],
                                       [1 - 0.05, 0.65]
                                       ])

        self.img_lane = None

    def callback(self, msg):
        try:
            np_arr = np.fromstring(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)

        self.img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # h = img_bgr.shape[0]
        # w = img_bgr.shape[1]
        lower_wlane = np.array([30, 0, 250])
        upper_wlane = np.array([40, 10, 255])

        img_wlane = cv2.inRange(img_hsv, lower_wlane, upper_wlane)

        lower_ylane = np.array([20, 240, 240])
        upper_ylane = np.array([45, 255, 255])

        img_ylane = cv2.inRange(img_hsv, lower_ylane, upper_ylane)

        self.img_lane = cv2.bitwise_or(img_wlane, img_ylane)

if __name__ == "__main__" :

    rp = rospkg.RosPack()

    currentPath = rp.get_path("lane_detection_example")

    with open(os.path.join(currentPath, "sensor/sensor_params.json"), "r") as fp:
        sensor_params = json.load(fp)

    params_cam = sensor_params["params_cam"]

    rospy.init_node("lane_detector", anonymous=True)

    image_parser = IMGParser()

    bev_op = BEVTransform(params_cam=params_cam)
    curve_learner = CURVEFit(order=1)
    # ctrller = purePursuit(lfd=0.8)
    # sline_detector = STOPLineEstimator()

    rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        # if image_parser.img_wlane is not None:

        #     img_warp = bev_op.warp_bev_img(image_parser.img_wlane)

        #     cv2.imshow("image_window", img_warp)

        #     cv2.waitKey(1)

        #     rate.sleep()

        if image_parser.img_lane is not None:

            img_warp = bev_op.warp_bev_img(image_parser.img_lane)
            lane_pts = bev_op.recon_lane_pts(image_parser.img_lane)

            # sline_detector.get_x_point(lane_pts)
            # sline_detector.estimate_dist(0.3)
            # sline_detector.visualize_dist()
            # sline_detector.pub_sline()

            x_pred, y_pred_l, y_pred_r = curve_learner.fit_curve(lane_pts)

            # curve_learner.write_path_msg(x_pred, y_pred_l, y_pred_r)
            # curve_learner.pub_path_msg()

            # ctrller.steering_angle(x_pred, y_pred_l, y_pred_r)
            # ctrller.pub_cmd()

            xyl, xyr = bev_op.project_lane2img(x_pred, y_pred_l, y_pred_r)

            img_warp1 = draw_lane_img(img_warp, xyl[:, 0].astype(np.int32),
                                                xyl[:, 1].astype(np.int32),
                                                xyr[:, 0].astype(np.int32),
                                                xyr[:, 1].astype(np.int32)
                                                )
            
            cv2.imshow("image_window", img_warp1)

            cv2.waitKey(1)

            rate.sleep()