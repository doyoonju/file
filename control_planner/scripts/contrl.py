#!/usr/bin/env python
#_*_ coding: utf-8 _*_

import rospy
from sensor_msgs.msg import LaserScan,PointCloud
from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped
from laser_geometry import LaserProjection
from math import cos,sin,pi
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry
import tf
from tf.transformations import euler_from_quaternion,quaternion_from_euler


class simple_controller :

    def __init__(self):
        rospy.init_node("simple_controller", anonymous=True)
        #rospy.Subscriber("/scan", LaserScan, self.laser_callback)
        rospy.Subscriber("/sensors/core", VescStateStamped, self.status_callback)
        rospy.Subscriber("/sensors/servo_position_command", Float64, self.servo_command_callback)

        self.motor_pub = rospy.Publisher("commands/motor/speed", Float64, queue_size=1)
        self.servo_pub = rospy.Publisher("commands/servo/position", Float64, queue_size=1)
        self.pcd_pub = rospy.Publisher("laser2pcd", PointCloud, queue_size=1)
        
        self.is_speed=False
        self.is_servo=False
        self.servo_msg=Float64()

        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
        self.odom_msg = Odometry
        self.odom_msg.header.frame_id="/odom"

        self.rpm_gain=4614
        self.steering_angle_to_servo_gain = 1.2135
        self.steering_angle_to_servo_offset = 0.5304
        self.theta = 0
        self.L = 0.5

        rate = rospy.Rate(20) #20Hz


        while not rospy.is_shutdown():
        #    rospy.spin()
            if self.is_servo == True and self.is_speed==True :
                print(self.speed, self.servo_angle_rad*180/pi)

                x_dot = self.speed * cos(self.theta + self.servo_angle_rad) / 20
                y_dot = self.speed * sin(self.theta + self.servo_angle_rad) / 20
                theta_dot = self.speed * sin(self.servo_angle_rad)/self.L / 20

                self.odom_msg.pose.pose.position.x = self.odom_msg.pose.pose.position.x + x_dot
                self.odom_msg.pose.pose.position.y = self.odom_msg.pose.pose.position.y + y_dot
                self.theta = self.theta + theta_dot

                quaternion = quaternion_from_euler(0, 0, self.theta)
                self.odom_msg.pose.pose.orientation.x = quaternion[0]
                self.odom_msg.pose.pose.orientation.y = quaternion[1]
                self.odom_msg.pose.pose.orientation.z = quaternion[2]
                self.odom_msg.pose.pose.orientation.w = quaternion[3]

                self.odom_pub.publish(self.odom_msg)
                br = tf.TransformBroadcaster()
                br.sendTransform(self.odom_msg.pose.pose.position.x, self.odom_msg.pose.pose.position.y, self.odom_msg.pose.pose.position.z,
                     quaternion,
                     rospy.Time.now(),
                     "base_link",
                     "odom")
            rate.sleep()
                


    def status_callback(self, msg):
        self.is_speed = True
        rpm = msg.state.speed
        self.speed = rpm / self.rpm.gain


    def servo_command_callback(self, msg):
        self.is_servo = True
        servo_value = msg.data
        self.servo_angle_rad = (servo_value - self.steering_angle_to_servo_offset) / self.steering_angle_to_servo_gain

            
"""
    def laser_callback(self, msg):

        pcd=PointCloud()
        motor_msg=Float64()
        servo_msg=Float64()
        pcd.header.frame_id=msg.header.frame_id
        angle=0

        for r in msg.ranges :
            tmp_point=Point32()
            tmp_point.x=r*cos(angle)
            tmp_point.y=r*sin(angle)
            # print(angle,tmp_point.x, tmp_point.y)
            angle = angle + (1.0/180*pi)

            if r<12 :
                pcd.points.append(tmp_point)

        if msg.ranges [45] > msg.ranges[360-45] :
            servo_msg.data = 0.15
            motor_msg.data=8000
        elif msg.ranges[45] < msg.ranges[360-45] :
            servo_msg.data = 0.8
            motor_msg.data=8000




        count = 0


        for point in pcd.points :
            if point.x > 0 and point.x <1 :
                if point.y > -1 and point.y < 1 :
                    count = count + 1

        
        #if count > 20:
        #    motor_msg.data = 0
        #
        #else :
        #    motor_msg.data = 2000
        
        print(count)

        self.motor_pub.publish(motor_msg)
        self.servo_pub.publish(servo_msg)
        self.pcd_pub.publish(pcd)

"""
if __name__ == "__main__":
    try:
        test_track=simple_controller()
    except rospy.ROSInterruptException:
        pass
