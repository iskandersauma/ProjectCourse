#!/usr/bin/env python

''' Input: A transform from some localization script
    Output: Broadcasts the transform
'''

import math
import rospy
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion,quaternion_from_euler
from geometry_msgs.msg import PoseStamped, TransformStamped
from aruco_msgs.msg import MarkerArray

class callback_handler(object):

    def __init__(self):
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.new_transform = TransformStamped()

        self.new_transform.header.frame_id = 'map'
        self.new_transform.header.stamp = rospy.Time.now()

        self.new_transform.child_frame_id = '/cf1/odom'

        self.new_transform.transform.translation.x = 0
        self.new_transform.transform.translation.y = 0
        self.new_transform.transform.translation.z = 0
        self.new_transform.transform.rotation.x = 0
        self.new_transform.transform.rotation.y = 0
        self.new_transform.transform.rotation.z = 0
        self.new_transform.transform.rotation.w = 1

    def transform_callback(self,tf):
        self.new_transform = tf

def main():
    rospy.init_node('localization_publisher')

    ch = callback_handler()
    sub_transform = rospy.Subscriber('localization_transform', TransformStamped, ch.transform_callback)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        ch.new_transform.header.stamp = rospy.Time.now()
        ch.broadcaster.sendTransform(ch.new_transform)
        rate.sleep()

if __name__ == '__main__':
    main()
