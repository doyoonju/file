#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import rospkg
from sensor_msgs.msg import LaserScan,PointCloud,Imu
from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped
from laser_geometry import LaserProjection
from math import cos,sin,pi,sqrt,pow
from geometry_msgs.msg import Point32,PoseStamped
from nav_msgs.msg import Odometry,Path
import numpy as np
import tf
from tf.transformations import euler_from_quaternion,quaternion_from_euler
from morai_msgs.msg import ObjectInfo

class velocityPlanning :
    def __init__(self,car_max_speed,road_friction):
        self.car_max_speed=car_max_speed
        self.road_friction=road_friction

    def curveBasedVelocity(self,global_path,point_num):
        out_vel_plan=[]
        for i in range(0,point_num):
            out_vel_plan.append(self.car_max_speed)
        for i in range(point_num,len(global_path.poses)-point_num):
            x_list=[]
            y_list=[]
            for box in  range(-point_num,point_num):
                x=global_path.poses[i+box].pose.position.x
                y=global_path.poses[i+box].pose.position.y
                x_list.append([-2*x,-2*y,1])
                y_list.append(-(x*x)-(y*y))
            
            x_matrix=np.array(x_list)
            y_matrix=np.array(y_list)
            x_trans=x_matrix.T

            a_matrix=np.linalg.inv(x_trans.dot(x_matrix)).dot(x_trans).dot(y_matrix)
            a=a_matrix[0]
            b=a_matrix[1]
            c=a_matrix[2]
            r=sqrt(a*a+b*b-c)
            v_max=sqrt(r*9.8*self.road_friction)*3.6  #0.7
            if v_max>self.car_max_speed :
                v_max=self.car_max_speed
            out_vel_plan.append(v_max)

        for i in range(len(global_path.poses)-point_num,len(global_path.poses)):
            out_vel_plan.append(self.car_max_speed)
        return out_vel_plan

class vaildObject :

    def __init__(self):
        
        self.vaild_stoplane_position=[
                                     [58.26,1180.09], 
                                     [85.56,1228.28]
                                     ] 

    def get_object(self,obj_msg):
        self.all_object=ObjectInfo()
        self.all_object=obj_msg


    def calc_vaild_obj(self,ego_pose):
        global_object_info=[]
        loal_object_info=[]
        if self.all_object.num_of_objects > 0:

            tmp_theta=ego_pose[2]
            tmp_translation=[ego_pose[0],ego_pose[1]]
            tmp_t=np.array([[cos(tmp_theta), -sin(tmp_theta),tmp_translation[0]],
                            [sin(tmp_theta),cos(tmp_theta),tmp_translation[1]],
                            [0,0,1]])
            tmp_det_t=np.array([[tmp_t[0][0],tmp_t[1][0],-(tmp_t[0][0]*tmp_translation[0]+tmp_t[1][0]*tmp_translation[1])   ],
                                [tmp_t[0][1],tmp_t[1][1],-(tmp_t[0][1]*tmp_translation[0]+tmp_t[1][1]*tmp_translation[1])   ],
                                [0,0,1]])

            for num in range(self.all_object.num_of_objects):
                global_result=np.array([[self.all_object.pose_x[num]],[self.all_object.pose_y[num]],[1]])
                local_result=tmp_det_t.dot(global_result)
                if local_result[0][0]>0 :
                    global_object_info.append([self.all_object.object_type[num],self.all_object.pose_x[num],self.all_object.pose_y[num],self.all_object.velocity[num]])
                    loal_object_info.append([self.all_object.object_type[num],local_result[0][0],local_result[1][0],self.all_object.velocity[num]])
            
            for num in range(len(self.vaild_stoplane_position)):
                global_result=np.array([[self.vaild_stoplane_position[num][0]],[self.vaild_stoplane_position[num][1]],[1]])
                local_result=tmp_det_t.dot(global_result)
                if local_result[0][0]>0 :
                    global_object_info.append([1,self.vaild_stoplane_position[num][0],self.vaild_stoplane_position[num][1],0])
                    loal_object_info.append([1,local_result[0][0],local_result[1][0],0])
                   

        return global_object_info,loal_object_info


    def acc(self,local_vaild_object,ego_val,target_vel):
        out_vel=target_vel

        if self.object[0] == True :
            print("ACC ON")
            front_vehicle = [local_vaild_object[self.object[1]][1],local_vaild_object[self.object[1]][2],local_vaild_object[self.object[1]][3]]
            time_gap=1
            default_space=2
            dis_safe=ego_val*time_gap+default_space
            dis_rel=sqrt(pow(front_vehicle[0],2)+pow(front_vehicle[1],2))-4.4
            print(dis_rel,dis_rel)
            vel_rel=(front_vehicle[2]-ego_val)*3.6
            v_gain=self.object_vel_gain
            x_errgain=self.object_dis_gain
            acceleration=velocityPlanning*v_gain-x_errgain*(dis_safe-dis_rel)

            acc_based_vel=ego_val*3.6+acceleration

            if acc_based_vel > target_vel :
                acc_based_vel=target_vel

            if dis_safe-dis_rel > 0:
                out_vel = acc_based_vel

            else :
                if acc_based_vel < target_vel :
                    out_vel=acc_based_vel

            

         







class path_pub_tf :

    def __init__(self):
        rospy.init_node('path_pub', anonymous=True)
        rospy.Subscriber("odom", Odometry, self.odom_callback)
        rospy.Subscriber("/obj_info", ObjectInfo, self.obj_callback)
        self.global_path_pub = rospy.Publisher('/global_path',Path, queue_size=1)
        self.local_path_pub = rospy.Publisher('/path',Path, queue_size=1)
        self.vel_pub = rospy.Publisher('/target_vel',Float64, queue_size=1)
        self.global_path_msg=Path()
        self.global_path_msg.header.frame_id='/map'
        
        self.is_odom=False
        self.local_path_size=50



        rospack=rospkg.RosPack()
        pkg_path=rospack.get_path('control_planner')
        full_path=pkg_path+'/path'+'/kcity.txt'
        self.f=open(full_path,'r')
        lines=self.f.readlines()
        for line in lines :
            tmp=line.split()
            read_pose=PoseStamped()
            read_pose.pose.position.x=float(tmp[0])
            read_pose.pose.position.y=float(tmp[1])
            read_pose.pose.orientation.w=1
            self.global_path_msg.poses.append(read_pose)
        
        self.f.close()


        vel_planner=velocityPlanning(60,0.15)
        vel_profile=vel_planner.curveBasedVelocity(self.global_path_msg,100)

        vaild_obj=vaildObject()

    

        rate = rospy.Rate(20) # 20hz
        while not rospy.is_shutdown():
   
            if self.is_odom == True :
                local_path_msg=Path()
                local_path_msg.header.frame_id='/map'
                
                x=self.x
                y=self.y
                min_dis=float('inf')
                current_waypoint=-1
                for i,waypoint in enumerate(self.global_path_msg.poses) :

                    distance=sqrt(pow(x-waypoint.pose.position.x,2)+pow(y-waypoint.pose.position.y,2))
                    if distance < min_dis :
                        min_dis=distance
                        current_waypoint=i

                
                if current_waypoint != -1 :
                    if current_waypoint + self.local_path_size < len(self.global_path_msg.poses):
                        for num in range(current_waypoint,current_waypoint + self.local_path_size ) :
                            tmp_pose=PoseStamped()
                            tmp_pose.pose.position.x=self.global_path_msg.poses[num].pose.position.x
                            tmp_pose.pose.position.y=self.global_path_msg.poses[num].pose.position.y
                            tmp_pose.pose.orientation.w=1
                            local_path_msg.poses.append(tmp_pose)
                    
                    else :
                        for num in range(current_waypoint,len(self.global_path_msg.poses) ) :
                            tmp_pose=PoseStamped()
                            tmp_pose.pose.position.x=self.global_path_msg.poses[num].pose.position.x
                            tmp_pose.pose.position.y=self.global_path_msg.poses[num].pose.position.y
                            tmp_pose.pose.orientation.w=1
                            local_path_msg.poses.append(tmp_pose)


                vel_msg=Float64()
                vel_msg.data=vel_profile[current_waypoint]
       
                

                vaild_obj.get_object(self.object_info_msg)
                global_vaild_object,local_vaild_object=vaild_obj.calc_vaild_obj([self.x,self.y,self.heading])
                for a in local_vaild_object :
                    print(a)
             





                self.global_path_pub.publish(self.global_path_msg)
                self.local_path_pub.publish(local_path_msg)
                self.vel_pub.publish(vel_msg)

            rate.sleep()



    

    def odom_callback(self,msg):
        self.is_odom=True
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        self.x=msg.pose.pose.position.x
        self.y=msg.pose.pose.position.y
        _,_,self.heading=euler_from_quaternion(odom_quaternion)

    def obj_callback(self,msg):
        self.object_info_msg=msg



if __name__ == '__main__':
    try:
        test_track=path_pub_tf()
    except rospy.ROSInterruptException:
        pass