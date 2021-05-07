#!/usr/bin/env python

''' Input: Marker sightings
    Output: A transform from map to odom to account for drift
'''

import math
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from tf.transformations import quaternion_from_matrix, quaternion_matrix
from geometry_msgs.msg import PoseStamped, TransformStamped
from aruco_msgs.msg import MarkerArray
from help_functions import *
from std_msgs.msg import String
from std_msgs.msg import UInt32

class localization(object):

    def __init__(self):
        # Listeners:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers:
        self.transform_publisher = rospy.Publisher('localization_transform', TransformStamped, queue_size=2)

        # Transform and weight buffers
        self.last_n_transforms = []
        self.weight_list = []

        # Initial transform
        self.next_transform = TransformStamped()
        self.next_transform.transform.rotation.w = 1

        # ----- CHANGE BUFFER LENGTH AND MARKER THRESHOLD HERE ----- #
        self.n = 10 #Buffer length
        self.marker_threshold = 2 #Marker threshold (maximum distance to marker)

    def marker_callback(self,data):
        markers = data.markers
        best_marker_dist = np.inf
        transform_available = False

        for marker in markers:

            marker_dist = marker_distance(marker)

            # Check if marker exists in map
            if not self.tf_buffer.can_transform('aruco/marker{}'.format(marker.id),'map',rospy.Time()):
                rospy.logwarn('No transform from aruco/marker{} to map'.format(marker.id))

            # Check if marker is reasonably close
            elif marker_dist > self.marker_threshold:
                rospy.logwarn('Marker %s is too far away', marker.id)

            else:
                transform_available = True

                # Choose the closest marker
                if marker_dist < best_marker_dist:
                    best_marker = marker
                    best_marker_dist = marker_dist


        if transform_available:
            rospy.logwarn("Chosen marker : %s ", best_marker.id)

            # Calculate the weight and transform for this marker
            marker_transform, tf_weight = self.calculate_transform(best_marker)

            if len(self.last_n_transforms) == 0:
                self.next_transform = marker_transform

            else:
                if len(self.last_n_transforms) > self.n -1:
                    self.last_n_transforms.pop(0)
                    self.weight_list.pop(0)

                temp_list = self.last_n_transforms + [marker_transform]
                temp_weights = self.weight_list + [tf_weight]

                # Average this transform with buffer transforms
                self.next_transform = average_transform(temp_list,temp_weights)

            # Put new transform in buffer
            self.last_n_transforms.append(self.next_transform)
            self.weight_list.append(tf_weight)

            # Publish transform
            rospy.loginfo('Next transform publishing : %s', self.next_transform)
            self.transform_publisher.publish(self.next_transform)


    def calculate_transform(self,marker):

        # Fetch position for marker in map
        p = self.tf_buffer.lookup_transform('map',
                                            'aruco/marker{}'.format(marker.id),
                                            rospy.Time(),
                                            rospy.Duration(5.0))

        # Fetch position for detected marker in odom
        q = self.tf_buffer.lookup_transform('cf1/odom',
                                            'aruco/detected{}'.format(marker.id),
                                            rospy.Time(),
                                            rospy.Duration(5.0))

        P = hcmatrix_from_transform(p)
        Q = hcmatrix_from_transform(q)

        Q_inv = np.linalg.inv(Q)

        # Calculate the transform
        T = np.dot(P,Q_inv)
        transform = transform_from_hcmatrix(T, 'map', 'cf1/odom')

        # Give the transform a weight depending on its adjustment,
        # i.e. how large the "jump" would be to adjust to the difference between map and detected marker
        # The larger "jump" the smaller weight

        weight = calculate_weight(p,q)

        return transform, weight


def main():
    rospy.init_node('localization')
    local = localization()
    aruco_subscriber = rospy.Subscriber('/aruco/markers', MarkerArray, local.marker_callback)
    rospy.spin()


if __name__ == "__main__":
    main()
