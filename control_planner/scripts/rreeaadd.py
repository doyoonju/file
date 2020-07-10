#!/usr/bin/env python
#_*_ coding: utf-8 _*_

import rospy
import rospkg
from sensor_msgs.msg import LaserScan,PointCloud, Imu
from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped
from laser_geometry import LaserProjection
from math import cos,sin,pi,sqrt,pow
from geometry_msgs.msg import Point32,PoseStamped
from nav_msgs.msg import Odometry,Path
import tf
from tf.transformations import euler_from_quaternion,quaternion_from_euler

class path_pub :

    def __init__(self):
        rospy.init_node("path_pub", anonymous=True)

        self.path_pub = rospy.Publisher("/path", Path, queue_size=1)
        self.path_msg = Path()
        self.path_msg.header.frame_id="/odom"

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("control_planner")
        full_path = pkg_path + "/path" + "/path_txt"
        self.f = open(full_path, "r")
        lines = self.f.readlines()

        for line in lines :
            tmp = line.split()
            read_pose = PoseStamped()
            read_pose.pose.position.x = float(tmp[0])
            read_pose.pose.position.y = float(tmp[1])
            read_pose.pose.orientation.w = 1
            self.path_msg.poses.append(read_pose)

        self.f.close()

        rate = rospy.Rate(10) #20Hz
        while not rospy.is_shutdown():
            self.path_pub.publish(self.path_msg)
            rate.sleep()



if __name__ == "__main__":
    try:
        test_track=path_pub()
    except rospy.ROSInterruptException:
        pass