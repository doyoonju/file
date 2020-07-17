#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import os, rospkg #os
import json

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridgeError
from std_msgs.msg import Float64
from utils import warp_image,BEVTransform,CURVEFit, draw_lane_img, purePursuit

class IMGParser:
    def __init__(self):

        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.callback)

        self.source_prop = np.float32([[0.05, 0.65],
                                       [0.5 - 0.15, 0.55],
                                       [0.5 + 0.15, 0.55],
                                       [1 - 0.05, 0.65]
                                       ])

        self.img_wlane = None

    def callback(self, msg):
        try:
            np_arr = np.fromstring(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            print(e)

        self.img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        lower_wlane = np.array([30, 0, 250])
        upper_wlane = np.array([40, 10, 255])

        self.img_wlane = cv2.inRange(img_hsv, lower_wlane, upper_wlane)

        # self.img_wlane = cv2.cvtColor(img_wlane, cv2.COLOR_GRAY2BGR)

        # img_warp = warp_image(img_wlane, self.source_prop)

        # img_concat = np.concatenate([img_bgr, img_hsv, img_wlane], axis=1) #1
        # #img_concat = np.concatenate([img_wlane, img_warp], axis=1)

        # #cv2.namedWindow("mouseRGB") #2

        # #cv2.imshow("mouseRGB", self.img_hsv) #2

        # #cv2.setMouseCallback("mouseRGB", self.mouseRGB) #2

        # #cv2.imshow("Image window", img_hsv) #1
        # cv2.imshow("Image window", img_warp)
        # cv2.waitKey(1)

    def warp_image(img, source_prop):

        image_size = (img.shape[1], img.shape[0])
        x = img.shape[1]
        y = img.shape[0]

        destination_points = np.float32([
            [0, y],
            [0, 0],
            [x, 0],
            [x, y]
        ])

        source_points = source_prop * np.float32([[x, y],[x, y], [x, y], [x, y]])

        perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)

        warped_img = cv2.warpPerspective(img, perspective_transform, image_size, flags = cv2.INTER_LINEAR)

        return warped_img
    """
    def mouseRGB(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            colorsB = self.img_hsv[y, x, 0]
            colorsG = self.img_hsv[y, x, 1]
            colorsR = self.img_hsv[y, x, 2]
            colors = self.img_hsv[y, x]
            print("Red: ", colorsR)
            print("Green: ", colorsG)
            print("Blue: ", colorsB)
            print("BGR format: ", colors)
            print("Coordinates of pixel: X:", x, "Y: ", y)
    """  
"""
if __name__ == "__main__":

    rospy.init_node("image_parser", anonymous=True)

    image_parser = IMGParser()

    rospy.spin()
"""

if __name__ == "__main__" :

    rp = rospkg.RosPack()

    currentPath = rp.get_path("lane_detection_example")

    with open(os.path.join(currentPath, "sensor/sensor_params.json"), "r") as fp:
        sensor_params = json.load(fp)

        params_cam = sensor_params["params_cam"]

        rospy.init_node("image_parser", anonymous=True)

        image_parser = IMGParser()

        bev_op = BEVTransform(params_cam=params_cam)
        curve_learner = CURVEFit(order=3)
        ctrller = purePursuit(lfd=0.8)
        rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        # if image_parser.img_wlane is not None:

        #     img_warp = bev_op.warp_bev_img(image_parser.img_wlane)

        #     cv2.imshow("image_window", img_warp)

        #     cv2.waitKey(1)

        #     rate.sleep()

        if image_parser.img_wlane is not None:

            img_warp = bev_op.warp_bev_img(image_parser.img_wlane)
            lane_pts = bev_op.recon_lane_pts(image_parser.img_wlane)

            x_pred, y_pred_l, y_pred_r = curve_learner.fit_curve(lane_pts)

            ctrller.steering_angle(x_pred, y_pred_l, y_pred_r)
            ctrller.pub_cmd()

            xyl, xyr = bev_op.project_lane2img(x_pred, y_pred_l, y_pred_r)

            img_warp1 = draw_lane_img(img_warp, xyl[:, 0].astype(np.int32),
                                                xyl[:, 1].astype(np.int32),
                                                xyr[:, 0].astype(np.int32),
                                                xyr[:, 1].astype(np.int32)
                                                )
            
            cv2.imshow("image_window", img_warp1)

            cv2.waitKey(1)

            rate.sleep()