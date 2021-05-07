#!/usr/bin/env python

''' Help-functions, mainly for localization.
    See each function for usage.

    Includes:
    marker_distance
    average_transform
    calculate_weight
    transform_from_hcmatrix
    hcmatrix_from_transform
    transform_distance
    normalize

'''

import math
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from tf.transformations import quaternion_from_matrix, quaternion_matrix, euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped, TransformStamped
from aruco_msgs.msg import MarkerArray

''' Input: Marker in some frame
    Output: Euclidean distance to marker in same frame
'''
def marker_distance(marker):
    x = marker.pose.pose.position.x
    y = marker.pose.pose.position.y
    z = marker.pose.pose.position.z

    distance = np.sqrt(x**2 + y**2 + z**2)

    return distance

''' Input: A list of transforms and a list of their respective weights
    Output: A new transform which is the weighted average of the input transforms
'''

def average_transform(list_of_transforms, list_of_weights):
    weight_sum = sum(list_of_weights)

    at= TransformStamped()
    at.header.stamp = rospy.Time.now()
    at.header.frame_id = list_of_transforms[0].header.frame_id
    at.child_frame_id = list_of_transforms[0].child_frame_id

    x_sum = 0
    y_sum = 0
    z_sum = 0

    Q = np.zeros((4,len(list_of_transforms)))

    for i in range(len(list_of_transforms)):
        x_sum += list_of_weights[i]*list_of_transforms[i].transform.translation.x
        y_sum += list_of_weights[i]*list_of_transforms[i].transform.translation.y
        z_sum += list_of_weights[i]*list_of_transforms[i].transform.translation.z

        Q[0,i] = list_of_weights[i]*list_of_transforms[i].transform.rotation.x
        Q[1,i] = list_of_weights[i]*list_of_transforms[i].transform.rotation.y
        Q[2,i] = list_of_weights[i]*list_of_transforms[i].transform.rotation.z
        Q[3,i] = list_of_weights[i]*list_of_transforms[i].transform.rotation.w

    at.transform.translation.x = x_sum/weight_sum
    at.transform.translation.y = y_sum/weight_sum
    at.transform.translation.z = z_sum/weight_sum

    w = np.divide(1,float(weight_sum))
    Q = w*Q

    eigenvalues, eigenvectors = np.linalg.eig(np.matmul(Q,np.transpose(Q)))

    averaged_quaternions = eigenvectors[:,eigenvalues.argmax()]
    averaged_quaternions = normalize(averaged_quaternions)


    at.transform.rotation.x = averaged_quaternions[0]
    at.transform.rotation.y = averaged_quaternions[1]
    at.transform.rotation.z = averaged_quaternions[2]
    at.transform.rotation.w = averaged_quaternions[3]

    return at

''' Input: Two transforms
    Output: The weight given to the transform pair, based on their angular and
            Euclidean distance

    Usage: See simple_localization

'''

def calculate_weight(t1, t2):

    # ----- SET MIN AND MAX WEIGHT HERE ----- #

    max_weight = 20
    min_weight = 1

    ad, ed = transform_distance(t1,t2)

    a_weight = max_weight*(1-2*ad)
    e_weight = max_weight*(1 - 9*ed**3)


    if a_weight < min_weight:
        a_weight = min_weight

    if e_weight < min_weight:
        e_weight = min_weight

    #Choose the minimum (least favourable) weight
    weight = min(a_weight, e_weight)

    return weight

''' Input: A transform
    Output: A matrix representation of the input transform in homogeneous coordinates
'''

def hcmatrix_from_transform(t):
    quats = [0, 0, 0, 0]
    quats[0] = t.transform.rotation.x
    quats[1] = t.transform.rotation.y
    quats[2] = t.transform.rotation.z
    quats[3] = t.transform.rotation.w

    hcm = quaternion_matrix(quats)

    hcm[0,3] = t.transform.translation.x
    hcm[1,3] = t.transform.translation.y
    hcm[2,3] = t.transform.translation.z

    return hcm

''' Input:  A matrix in homogeneous coordinates, header and child frame for
            requested transformation.
    Output: Transform representation of the homogeneous coordinate matrix.
'''

def transform_from_hcmatrix(hcm,header_frame,child_frame):
    quats = quaternion_from_matrix(hcm)

    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = header_frame
    t.child_frame_id = child_frame

    t.transform.translation.x = hcm[0,3]
    t.transform.translation.y = hcm[1,3]
    t.transform.translation.z = hcm[2,3]

    t.transform.rotation.x = quats[0]
    t.transform.rotation.y = quats[1]
    t.transform.rotation.z = quats[2]
    t.transform.rotation.w = quats[3]

    return t

''' Input:  Two transforms
    Output: The angular and euclidean distance between input transforms
'''

def transform_distance(transform1, transform2):
    quats1 = [transform1.transform.rotation.x,
                transform1.transform.rotation.y,
                transform1.transform.rotation.z,
                transform1.transform.rotation.w]
    quats2 = [transform2.transform.rotation.x,
                transform2.transform.rotation.y,
                transform2.transform.rotation.z,
                transform2.transform.rotation.w]

    scalar_product = np.dot(quats1,quats2)

    angular_distance = 1 - scalar_product**2

    delta_x = transform1.transform.translation.x-transform2.transform.translation.x
    delta_y = transform1.transform.translation.y-transform2.transform.translation.y
    delta_z = transform1.transform.translation.z-transform2.transform.translation.z
    euclidean_distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    return angular_distance, euclidean_distance

''' Input: Vector to be normalized
    Output: Normalized vector
'''

def normalize(v):
    norm = np.linalg.norm(v)
    result = np.divide(v,float(norm))
    return result
