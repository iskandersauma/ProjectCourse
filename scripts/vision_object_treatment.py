#!/usr/bin/env python
from __future__ import print_function

import math
import rospy
import cv2
import sys
import numpy as np
#from scipy.ndimage import label, generate_binary_structure
from std_msgs.msg import String
from pras_project.msg import ObjectPose
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped



class objectPostTreatment:

    def __init__(self,path):

        #INPUT
        self.object_sub = rospy.Subscriber("/vision/object_pose", ObjectPose, self.callback_ObjectPose)
        self.mission_state_sub = rospy.Subscriber("/mission_planning/end_of_mission", String, self.callback_EndMission)

        #FILTER PARAMETERS
        self.min_dist_between_same = 2 #Two object of the same type cannot be too close
        self.min_dist_between_diff = 0.25 #Two objects cannot intersect
        self.min_weight = 10 #Minimum weight to consider an object

        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buf   = tf2_ros.Buffer()
        self.tf_lstn  = tf2_ros.TransformListener(self.tf_buf)
        self.path = path

        self.sign_names = ["airport",
            "residential",
            "dangerous_curve_left",
            "dangerous_curve_right",
            "junction",
            "road_narrows_from_left",
            "road_narrows_from_right",
            "circulation_warning",
            "follow_left",
            "follow_right",
            "no_bicycle",
            "no_heavy_truck",
            "stop",
            "no_parking",
            "no_stopping_and_parking"]

        self.objects_seen = np.zeros((10,15,7))
        self.objects_number = np.zeros((15,1), dtype=int)
        self.weights = np.zeros((10,15,1))


    # Creating the txt file containing all the poses on call of this topic :
    # rostopic pub /vision/end_of_mission std_msgs/String "go_final"
    def callback_EndMission(self,msg):
        #Create the file
        if msg.data == "go_final":

            list_incorrect = self.filter_obj_seen()

            rospy.loginfo("*******FINAL PROCEDURE********\n Creating file with object(s) position(s)..")

            file = open(self.path+"pose.txt","w")
            for i_obj in range(15):
                for j in range(self.objects_number[i_obj]):

                    if (j,i_obj) not in list_incorrect:

                        rospy.loginfo("Object : %s ",self.sign_names[i_obj])
                        rospy.loginfo("Pose : %s ",self.objects_seen[j][i_obj])
                        rospy.loginfo("W : %s ",self.weights[j][i_obj][0])

                        str_obj = self.sign_names[i_obj] +'\t'+ '\t'.join(str(e) for e in self.objects_seen[j][i_obj])+'\n'
                        file.write(str_obj)
                    else:
                        rospy.logwarn("Object : %s ",self.sign_names[i_obj])
                        rospy.logwarn("Pose : %s ",self.objects_seen[j][i_obj])
                        rospy.logwarn("W : %s ",self.weights[j][i_obj][0])


            file.close()
            rospy.loginfo("Done.")
        return


    #Last call function, in order to sort a little bit all the data
    def filter_obj_seen(self):

        #List of objects that are not relevant
        to_delete = []
        keep_info = []
        #Checkink object after object
        for obj_id in range(15):
            for k in range(self.objects_number[obj_id][0]):

                w1 = self.weights[k][obj_id][0]
                my_obj = self.objects_seen[k][obj_id]

                if w1 > self.min_weight or self.objects_number[obj_id][0] == 1 :

                    #Checking the distance with the other objects
                    for other_obj in range(obj_id+1,15):
                        for k_bis in range(self.objects_number[other_obj][0]):
                            next_obj = self.objects_seen[k_bis][other_obj]
                            d = self.get_dist(my_obj,next_obj)
                            #If two objects are too close, one of them is not good
                            if d < self.min_dist_between_diff:

                                w2 = self.weights[k_bis][other_obj][0]
                                #Keeping the object with bigger weight
                                if w1 > w2:
                                    to_delete.append((k_bis,other_obj))
                                    keep_info.append((k,obj_id))
                                else:
                                    to_delete.append((k,obj_id))
                                    keep_info.append((k_bis,other_obj))

                #The object has not been seen enough time
                else:
                    to_delete.append((k,obj_id))

        rospy.logwarn("Some objects are deleted with distance filter : \n%s",to_delete)

        return to_delete



    def callback_ObjectPose(self,object_pose):

        # TODO #
        # Save all the images in variables in order to do post treatment

        ref_id = self.sign_names.index(object_pose.name)

        object_odom = self.get_pose_from_cam_to_map(object_pose.pose)
        if object_odom is None:
            rospy.logwarn_throttle(0.5,"Problem in the transform to map !!!")
        else :
            self.update_positions(object_odom,ref_id,object_pose.confidence,object_pose.name)
            #self.publish_image_TF(object_odom,object_pose.name)
        return


    #Calculate the distance between two poses
    def get_dist(self,p1,p2):

        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)

        return dist



    def update_positions(self,obj_pose, ref_id, conf, name):

        vect_pose = np.array([obj_pose.pose.position.x,
            obj_pose.pose.position.y,
            obj_pose.pose.position.z,
            obj_pose.pose.orientation.x,
            obj_pose.pose.orientation.y,
            obj_pose.pose.orientation.z,
            obj_pose.pose.orientation.w],dtype = "float32")

        #Check with the other object of same type
        if self.objects_number[ref_id] > 0:

            dists = [self.get_dist(self.objects_seen[k][ref_id],vect_pose) for k in range (self.objects_number[ref_id][0]) ]

            if min(dists) > self.min_dist_between_same and self.objects_number[ref_id][0] < 10:
                k_min = self.objects_number[ref_id][0]
                self.objects_number[ref_id][0] = self.objects_number[ref_id][0] + 1
            else :
                k_min = np.argmin(dists)

        #First time we see this object
        else:
            k_min = 0
            self.objects_number[ref_id][0] = 1

        up_pose = self.objects_seen[k_min][ref_id]*self.weights[k_min][ref_id] + vect_pose*conf

        #New weights
        self.weights[k_min][ref_id] = self.weights[k_min][ref_id] + conf
        #New pose
        self.objects_seen[k_min][ref_id] = up_pose/self.weights[k_min][ref_id]

        self.publish_image_TF(obj_pose, name+str(k_min))

        return


    #Calculate the pose of the object in the world frame
    def get_pose_from_cam_to_map(self, pose):

        if not self.tf_buf.can_transform(pose.header.frame_id, 'map', pose.header.stamp, rospy.Duration(2)):
            rospy.logwarn_throttle(5.0, 'No transform from %s to map' % pose.header.frame_id)
            return None

        #Transforming to map
        object_odom = self.tf_buf.transform(pose, 'map')

        return object_odom


    #Publishing the TF of the object / for debugging
    def publish_image_TF(self,pose,name):

        t = TransformStamped()
        t.header.frame_id = "cf1/camera_link"
        t.header.stamp = pose.header.stamp

        t.child_frame_id = '/Object/' + name
        t.transform.translation = pose.pose.position
        t.transform.rotation = pose.pose.orientation

        #self.broadcaster.sendTransform(t)

        return



def main(argv=sys.argv):

    rospy.init_node('object_treatment')
    args = rospy.myargv(argv=argv)
    ic = objectPostTreatment(args[1])

    print("POST TREATMENT --> running...")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
