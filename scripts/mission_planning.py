#!/usr/bin/env python

from __future__ import print_function

import math
import rospy
import sys
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import UInt32
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped, Vector3
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import json



class mission_planner:

    def __init__(self,gates_poses):

        self._gates_poses = gates_poses

        #Output : Publishing the different poses trough this topic
        self.pub_goal = rospy.Publisher('mission_planning/next_goal', PoseStamped, queue_size=10)
        self.mission_state_pub = rospy.Publisher('/mission_planning/end_of_mission', String, queue_size=2)
        self.mission_planning_order_pub = rospy.Publisher('/mission_planning/order', String, queue_size=2)

        #Input :
        self.sub_goal = rospy.Subscriber('path_planning/state', String, self.callback_reached)
        self.sub_pose = rospy.Subscriber('/cf1/pose',PoseStamped,self.callback_position)
        self.sub_battery_level = rospy.Subscriber('/cf1/battery',Float32,self.callback_battery)

        self._state_planning = "waiting"
        self._state_peter = "landed"
        self._state_battery = "ok"
        self._current_pose = PoseStamped()
        self._current_gate = 0
        self._critical_battery_level = 3.0
        self._buffer_batt = []


    def callback_battery(self,msg):

        if len (self._buffer_batt) >= 10:
            self._buffer_batt.pop(0)
        self._buffer_batt.append(msg.data)

        if np.mean(self._buffer_batt) < self._critical_battery_level:
            self._state_battery = 'low'
            self.mission_planning_order_pub.publish("landing")
        return

    ### Updating the position in real time
    def callback_position(self,pose):
        #Getting the current pose
        self._current_pose = pose
        return


    # Actualize the state of peter depending on the response of path planning
    def callback_reached(self,msg):

        if msg.data == 'done':

            if self._state_planning == 'running':
                self._state_planning = "waiting"

            if self._state_peter == 'flying_free':
                self._state_peter = 'flying_to_gate'

            elif self._state_peter == 'in_error':
                self._state_peter = 'flying_to_gate'

            elif self._state_peter == 'flying_to_gate':
                self._state_peter = 'flying_in_gate'

            elif self._state_peter == 'flying_in_gate':
                self._state_peter = 'flying_to_gate'
                self._current_gate = self._current_gate + 1
                if self._current_gate*2 == len(self._gates_poses):
                    self._state_peter = 'finished'

            elif self._state_peter == 'exploring':
                self._state_peter = 'exploring'
                if len(self._exploration_marker_list) == 0:
                    self._state_peter = 'finished'

        if msg.data == 'path_fail':
            self._state_planning = 'running'
            self._state_peter = 'in_error'

        #Calling state machine to know what to do now
        self.state_machine()

        return



    # Computing next state from the current one
    def state_machine(self):

        if self._state_peter == 'landed' and self._state_planning == 'waiting' and self._state_battery == 'ok':
            rospy.loginfo("MISSION PLANNING : Ready for take off !")

            self.mission_planning_order_pub.publish("gate_altitude")
            rospy.sleep(0.5)

            self._state_peter = 'flying_free'
            self._state_planning = 'running'
            self.pub_goal.publish(self._current_pose)

            return


        if self._state_peter == 'flying_to_gate' and self._state_planning == 'waiting' and self._state_battery == 'ok':
            rospy.loginfo("MISSION PLANNING : Publishing next gate approach")
            self.mission_planning_order_pub.publish("free_altitude")
            rospy.sleep(0.5)

            self._state_planning = 'running'
            self.pub_goal.publish(self._gates_poses[self._current_gate*2])
            return


        if self._state_peter == 'flying_in_gate' and self._state_planning == 'waiting' and self._state_battery == 'ok':
            rospy.loginfo("MISSION PLANNING : Going trough the gate !")
            self.mission_planning_order_pub.publish("gate_altitude")
            rospy.sleep(1)

            self._state_planning = 'running'
            self.pub_goal.publish(self._gates_poses[self._current_gate*2+1])
            self.mission_state_pub.publish("go_final")


            return

        if self._state_peter == 'exploring' and self._state_planning == 'waiting' and self._state_battery == 'ok':
            rospy.loginfo(" MISSION PLANNING : Exploration mode !")
            self.mission_planning_order_pub.publish("free_altitude")
            rospy.sleep(0.5)
            self._state_planning = 'running'
            self.pub_goal.publish(self._exploration_marker_list[0][0])
            self._exploration_marker_list.pop(0)
            return


        if self._state_peter == 'finished' and self._state_planning == 'waiting' and self._state_battery == 'ok':
            rospy.loginfo("***************************************")
            rospy.loginfo(" MISSION PLANNING : Mission completed !")
            rospy.loginfo("***************************************")
            self.mission_state_pub.publish("go_final")
            self.mission_planning_order_pub.publish("landing")
            return


        if self._state_battery == 'low':
            rospy.loginfo("***************************************")
            rospy.loginfo(" LOW BATTERY ! Aborting mission ! ")
            rospy.loginfo("***************************************")
            self.mission_state_pub.publish("go_final")
            self.mission_planning_order_pub.publish("landing")

            return


        return

    def shutdown_protocol(self):
        print("Publishing final protocol...")
        self.mission_state_pub.publish("go_final")
        print("Shutting down")



#Create a very simple pose with x, y and yaw angle
def create_simple_poseStamped_XYT(X,Y,T):
    P = PoseStamped()
    P.header.stamp = rospy.Time.now()
    P.header.frame_id = 'map'
    P.pose.position.x = X
    P.pose.position.y = Y
    P.pose.position.z = 0
    roll, pitch, yaw = 0,0,T
    (P.pose.orientation.x,
    P.pose.orientation.y,
    P.pose.orientation.z,
    P.pose.orientation.w) = quaternion_from_euler(roll,pitch,yaw, axes='sxyz')

    return P


def poses_from_gates(m):

    offset_gates = 0.25

    Xc = np.array([m['position'][0], m['position'][1]])
    theta = m['heading']

    Offset = np.array([offset_gates*np.cos(math.radians(theta)), offset_gates*np.sin(math.radians(theta))])

    point_before = Xc - 2*Offset
    point_after = Xc + Offset

    # Creating two poseStamped object of the two positions
    P1 = create_simple_poseStamped_XYT(point_before[0],point_before[1],math.radians(theta))
    P2 = create_simple_poseStamped_XYT(point_after[0],point_after[1],math.radians(theta))

    return P1, P2



def main(argv=sys.argv):

    rospy.init_node('mission_planning')

    # Let ROS filter through the arguments
    args = rospy.myargv(argv=argv)
    # Load world JSON
    with open(args[1], 'rb') as f:
        world = json.load(f)

    # Create a list of all the position of the gates
    gates_poses = [[poses_from_gates(m)[0],poses_from_gates(m)[1]] for m in world['gates']]

    flat_list_gates = [item for sublist in gates_poses for item in sublist]


    ic = mission_planner(flat_list_gates)

    print("MISSION PLANNING --> running...")

    #Initializinf take off
    rospy.sleep(2)
    ic.state_machine()

    rospy.spin()

    rospy.on_shutdown(ic.shutdown_protocol())


if __name__ == '__main__':
    main(sys.argv)
