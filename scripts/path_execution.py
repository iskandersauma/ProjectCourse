#!/usr/bin/env python
from __future__ import print_function

import rospy
import tf2_geometry_msgs
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from crazyflie_driver.msg import Position
import path_planning
from matplotlib import pyplot as plt
import json
import numpy as np
import math
import sys

class executionner:

    def __init__(self, grid_gates, grid_04, grid_1, grid_x, grid_y, grid_max, grid_min, fig,ax):

        #INPUT : The pose of the drone and the final goal
        self.sub_pose = rospy.Subscriber('/cf1/pose',PoseStamped,self.callback_position)
        self.sub_goal = rospy.Subscriber('mission_planning/next_goal', PoseStamped, self.callback_goal_received)
        self.mission_order = rospy.Subscriber('/mission_planning/order', String, self.callback_mission_order)

        #OUTPUT : Keypoints to reach the goal
        self.pub_cmd  = rospy.Publisher('/cf1/cmd_position', Position, queue_size=2)
        self.pub_end = rospy.Publisher('path_planning/state', String, queue_size=2)

        #Occupancy grids
        self._grid_gates = grid_gates
        self._grid_1 = grid_1
        self._grid_04 = grid_04

        self._grid_x = grid_x
        self._grid_y = grid_y
        self._grid_max = grid_max
        self._grid_min = grid_min

        self._my_fig = fig
        self._my_ax = ax

        self.path = []
        self.altitude = 0.33
        self._current_pose = PoseStamped()
        self._current_command = None
        self._poseStamped_command_MAP = None
        self.on_position = False
        self._waiting_for_path = True
        self._tolerance = 0.10

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self._planning_is_free = True

    def callback_mission_order(self,msg):
        if msg.data == "gate_altitude":
            self._planning_is_free = False
        if msg.data == "free_altitude":
            self._planning_is_free = True
        if msg.data == "landing":
            self._current_command = None
            self._poseStamped_command_MAP = None
        return


    def publish_command(self):

        if self.set_new_command_in_map is not None:
            #Transforming at current time
            self._poseStamped_command_MAP.header.stamp = rospy.Time.now()

            if not self.tf_buffer.can_transform( self._poseStamped_command_MAP.header.frame_id, 'cf1/odom', self._poseStamped_command_MAP.header.stamp, rospy.Duration(1)):
                rospy.logwarn_throttle(5.0, 'SHIT ! Cannot convert the command into odom frame !!!')
                return None

            #Transforming the command from map to odom
            cmd_in_odom = self.tf_buffer.transform(self._poseStamped_command_MAP, 'cf1/odom')

            (roll,pitch,yaw) = euler_from_quaternion([cmd_in_odom.pose.orientation.x,
                cmd_in_odom.pose.orientation.y,
                cmd_in_odom.pose.orientation.z,
                cmd_in_odom.pose.orientation.w], axes='sxyz')


            # Creating the Position() object for the command
            self._current_command = Position()
            self._current_command.header.stamp = rospy.Time.now()
            self._current_command.header.frame_id = 'cf1/odom'
            self._current_command.x = cmd_in_odom.pose.position.x
            self._current_command.y = cmd_in_odom.pose.position.y
            self._current_command.z = self.altitude
            self._current_command.yaw = math.degrees(yaw)

            #Publishing the correct command
            self.pub_cmd.publish(self._current_command)

        return




    #Updating the position in real time
    def callback_position(self,pose):

        #Getting the current pose
        self._current_pose = pose

        if self._poseStamped_command_MAP is not None:
            self.publish_command()

        if not self._waiting_for_path:

            #Check if we have reached the command
            self.check_reached()
            #Defining a command if None is existing yet

            if len(self.path) > 0 and self._current_command is None and self._poseStamped_command_MAP is None:
                self.set_new_command_in_map(self.path[0][0],self.path[0][1],self.path[0][2])


            #On position : sending the new keypoint as a command
            elif len(self.path) > 0 and self.on_position:
                rospy.logwarn("KeyPoint reached !")
                #rospy.sleep(0.1)
                self.on_position = False
                self.set_new_command_in_map(self.path[0][0],self.path[0][1],self.path[0][2])

                self.path.pop(0)

            #On position and path is done
            elif len(self.path) == 0:
                rospy.loginfo("Path completed ! Waiting for new point.")
                rospy.sleep(0.2)
                self.pub_end.publish('done')
                self._waiting_for_path = True

        return


    ### Function called every time a goal is sent by mission planning ###
    def callback_goal_received(self,cmd):

        ### Creating starting and ending points for path planning : both must be in map frame !

        ### If there is already a command, we start from this one, and create the finish with the other one
        if self._current_command is not None:
            # START : Initializing the search from the current position
            (roll,pitch,yaw) = euler_from_quaternion([self._poseStamped_command_MAP.pose.orientation.x,
                self._poseStamped_command_MAP.pose.orientation.y,
                self._poseStamped_command_MAP.pose.orientation.z,
                self._poseStamped_command_MAP.pose.orientation.w], axes='sxyz')
            start = [(self._poseStamped_command_MAP.pose.position.x,self._poseStamped_command_MAP.pose.position.y,math.degrees(yaw))]
            # FINISH : Creating the finish pose from received goal
            (roll,pitch,yaw) = euler_from_quaternion([cmd.pose.orientation.x,
                cmd.pose.orientation.y,
                cmd.pose.orientation.z,
                cmd.pose.orientation.w], axes='sxyz')
            finish = [(cmd.pose.position.x,cmd.pose.position.y,math.degrees(yaw))]

        ### If no current command : take off. Start and Finish at the same position
        else:
            # In this case, the cmd is equal to the drone pose in odom frame
            if not self.tf_buffer.can_transform( cmd.header.frame_id, 'map',cmd.header.stamp, rospy.Duration(0.5)):
                rospy.logwarn_throttle(5.0, 'SHIT ! First plan will be a mess, no transform from %s to map' % cmd.header.frame_id)
                #return None
            #Transforming the current pose to map
            cmd = self.tf_buffer.transform(cmd, 'map')
            (roll,pitch,yaw) = euler_from_quaternion([cmd.pose.orientation.x,
                cmd.pose.orientation.y,
                cmd.pose.orientation.z,
                cmd.pose.orientation.w], axes='sxyz')
            start = [(cmd.pose.position.x,cmd.pose.position.y,math.degrees(yaw))]
            finish = [(cmd.pose.position.x,cmd.pose.position.y,math.degrees(yaw))]


        # Calling for path planning
        self.call_for_path_planning(start,finish)

        return


    def call_for_path_planning(self,start,finish):

        # Log debug
        rospy.loginfo("PATH PLANNING STARTED BETWEEN : \n\tStart  = %s \n\tFinish = %s",start,finish)

        ### Calling path planning with one, or several grids
        if self._planning_is_free:
            self.path, self.altitude = path_planning.plan_rout(start, finish, [self._grid_04,self._grid_1], self._grid_x, self._grid_y, self._grid_max, self._grid_min)
        else:
            self.path, self.altitude = path_planning.plan_rout(start, finish, [self._grid_gates], self._grid_x, self._grid_y, self._grid_max, self._grid_min)


        ### Looking for the results of the planning
        #Path planning failed : the current position is probably too close to a wall
        if self.path is None:
            rospy.logwarn("Path planning failed ! Trying to get away from walls")
            self.pub_end.publish('path_fail')
            # creating a very small path just to move away from the current position
            self.path , self.altitude = self.move_away_from_walls()
        #Path planning worked
        else:
            rospy.loginfo("Path OK ! GO")
            rospy.loginfo(self.path)


        ### Plot for debug
        X = []
        Y = []
        for e in self.path:
            X.append(e[0])
            Y.append(e[1])
        self._my_ax.plot(X,Y)
        self._my_fig.savefig('bonjour.jpg')


        self._waiting_for_path = False

        return

    ### Function computing the number of collision between two points ###
    def collision_on_line(self,x_0,y_0,x_1,y_1):

        x_0 = float(x_0*100) + float(self._grid_x)
        y_0 = float(y_0*100) + float(self._grid_y)
        x_1 = float(x_1*100) + float(self._grid_x)
        y_1 = float(y_1*100) + float(self._grid_y)

        if int(x_0-x_1) == 0 and int(y_0-y_1) == 0:
            return 0
        n = 0
        #Computing linear function between the two points
        if abs(x_0-x_1) > abs(y_0-y_1):
            m = (y_0-y_1)/(x_0-x_1)
            p = y_0 - m*x_0
            for x in range(int(x_0),int(x_1),int(np.sign(x_1-x_0))):
                y = int(m*x+p)
                if x < len(self._grid_gates) and y < len(self._grid_gates[0]):
                    if self._grid_gates[x][y] == ['occupied']:
                        n = n + 1
                else:
                    n = n+1
        else:
            m = (x_0-x_1)/(y_0-y_1)
            p = x_0 - m*y_0
            for y in range(int(y_0),int(y_1),int(np.sign(y_1-y_0))):
                x = int(m*y+p)
                if x < len(self._grid_gates) and y < len(self._grid_gates[0]):
                    if self._grid_gates[x][y] == ['occupied']:
                        n = n + 1
                    else:
                        n = n+1
        return n

    ### Function that move the drone away from wall by small step ###
    def move_away_from_walls(self):


        if not self.tf_buffer.can_transform( self._current_pose.header.frame_id, 'map',self._current_pose.header.stamp, rospy.Duration(0.5)):
                rospy.logerr('SHIT ! Cannot convert the pose into map frame when trying to move away from walls')
                #return None

        #Transforming the current pose to map
        pose_in_odom = self.tf_buffer.transform(self._current_pose, 'map')

        step_size = 0.1
        x_0 = pose_in_odom.pose.position.x
        y_0 = pose_in_odom.pose.position.y

        #Directions to try the collisions on
    	yaw_try = [-3*math.pi/4, -math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]
        #Computing the number of collisions
        coll = [ self.collision_on_line(x_0,y_0,x_0 + step_size*math.cos(yaw), y_0 + step_size*math.sin(yaw)) for yaw in yaw_try]
        #Computing the average yaw weighted with the number of collision
        bad_yaw = 0
        total_coll = sum(coll)
        for y in range(len(yaw_try)):
            bad_yaw = bad_yaw + yaw_try[y]*coll[y]
        #Going in the opposite direction
        yaw = -bad_yaw/total_coll


        return [(x_0 + step_size*math.cos(yaw), y_0 + step_size*math.sin(yaw), yaw)], self._current_pose.pose.position.z



    ### Function to create a goal into the map frame
    def set_new_command_in_map(self,x,y,yaw):

        # Goal PoseStamped into the MAP frame !
        self._poseStamped_command_MAP = PoseStamped()
        self._poseStamped_command_MAP.header.frame_id = 'map'
        self._poseStamped_command_MAP.header.stamp = rospy.Time.now()
        self._poseStamped_command_MAP.pose.position.x = x
        self._poseStamped_command_MAP.pose.position.y = y
        self._poseStamped_command_MAP.pose.position.z = self.altitude
        (self._poseStamped_command_MAP.pose.orientation.x,
        self._poseStamped_command_MAP.pose.orientation.y,
        self._poseStamped_command_MAP.pose.orientation.z,
        self._poseStamped_command_MAP.pose.orientation.w) = quaternion_from_euler(0,0,math.radians(yaw), axes='sxyz')

        return



    ### Function that update the on_position parameter if the pose matches with the command
    def check_reached(self):
        if self._current_command == None:
            self.on_position = False
            return False

        dx = self._current_pose.pose.position.x - self._current_command.x
        dy = self._current_pose.pose.position.y - self._current_command.y
        dz = self._current_pose.pose.position.z - self._current_command.z

        dist = np.sqrt(dx**2 + dy**2 + dz**2)


        # TODO
        ## Add some yaw control ?



        if dist < self._tolerance:
            self.on_position = True

        return



###################################
### CREATING THE OCCUPANCY GRID ###
###################################

def create_grid(maximum,minimum):
    size_x = maximum[0] - minimum[0]
    size_y = maximum[1] - minimum[1]
    grid = ['']*(size_x*100+1)
    for i in range(len(grid)):
	    grid[i] = ['free']*(size_y*100+1)
    return grid

def add_wall_to_grid(grid,walls,x_c,y_c,fly_altitude):
    start = walls['plane']['start']
    stop = walls['plane']['stop']

    if fly_altitude + 0.1 < start[2] or fly_altitude - 0.1 > stop[2]:
        return

    angle = math.atan2((stop[1]- start[1]),(stop[0]-start[0]))
    wall_size = math.sqrt((stop[1]- start[1])**2 + (stop[0]-start[0])**2)
    steps = int(wall_size*100)
    for i in range(-10,steps + 11):
        for j in range(-10,11):
            x = start[0]*100 + i*math.cos(angle) + j*math.cos(angle + math.pi/2)
            x_est = int(x)
            y = start[1]*100 + i*math.sin(angle) + j*math.sin(angle + math.pi/2)
            y_est = int(y)
            #print([x_est + x_c,y_est + y_c])
            if x_est + x_c >=0 and y_est + y_c >= 0 and x_est + x_c < len(grid) and y_est + y_c <  len(grid[0]):
                grid[x_est + x_c][y_est + y_c] = ['occupied']
    return



def add_gate_to_grid(grid,gate,x_c,y_c,open_gates):
    start = gate['position']
    angle = math.radians(gate['heading'])

    #Adding occupoied cells in the middle of the gate
    if not open_gates:
        for i in range(0,16):
            for j in range(-5,6):
                x = start[0]*100 + i*math.cos(angle + math.pi/2) + j*math.cos(angle)
                x_est = int(x)
                y = start[1]*100 + i*math.sin(angle + math.pi/2) + j*math.sin(angle)
                y_est = int(y)
                grid[x_est + x_c][y_est + y_c] = ['occupied']

                x = start[0]*100 - i*math.cos(angle + math.pi/2) - j*math.cos(angle)
                x_est = int(x)
                y = start[1]*100 - i*math.sin(angle + math.pi/2) - j*math.sin(angle)
                y_est = int(y)
                grid[x_est + x_c][y_est + y_c] = ['occupied']

    for i in range(16,41):
        for j in range(-15,16):
            x = start[0]*100 + i*math.cos(angle + math.pi/2) + j*math.cos(angle)
            x_est = int(x)
            y = start[1]*100 + i*math.sin(angle + math.pi/2) + j*math.sin(angle)
            y_est = int(y)
            grid[x_est + x_c][y_est + y_c] = ['occupied']

            x = start[0]*100 - i*math.cos(angle + math.pi/2) - j*math.cos(angle)
            x_est = int(x)
            y = start[1]*100 - i*math.sin(angle + math.pi/2) - j*math.sin(angle)
            y_est = int(y)
            grid[x_est + x_c][y_est + y_c] = ['occupied']


def get_center(maximum, minimum):
    x_c = - minimum[0]*100
    y_c = - minimum[1]*100
    return x_c, y_c


def make_occupancy_grid(file_path, fly_altitude, open_gates):

    # Load world JSON
    with open(file_path, 'rb') as f:
	    world = json.load(f)

    maximum = world['airspace']['max']
    minimum = world['airspace']['min']
    maxi = (maximum[0],maximum[1])
    mini = (minimum[0],minimum[1])
    grid = create_grid(maximum,minimum)
    x_c, y_c = get_center(maximum, minimum)
    for m in world['walls']:
	add_wall_to_grid(grid,m,x_c,y_c,fly_altitude)
    for m in world['gates']:
    	add_gate_to_grid(grid,m,x_c,y_c,open_gates)
    return grid, x_c, y_c, maxi, mini

def create_map_png(grid,mini,maxi,x_c, y_c):
    size_x = maxi[0] - mini[0]
    size_y = maxi[1] - mini[1]
    free_X = []
    free_Y = []
    occ_X = []
    occ_Y = []
    for i in range (size_x*100-1):
        for j in range (size_y*100-1):
            if grid[i][j] != "free":
                occ_X.append((i - x_c)/100.)
                occ_Y.append((j - y_c)/100.)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(occ_X,occ_Y,'r.')
    ax1.plot(free_X,free_Y,'b.')
    ax1.axis('equal')
    fig.savefig('my_occ_map.png')
    return fig,ax1


def main(argv=sys.argv):
    rospy.init_node('path_execution')

    # Let ROS filter through the arguments
    args = rospy.myargv(argv=argv)

    grid_gates, x_c, y_c, maximum, minimum = make_occupancy_grid(args[1], 0.33, True)
    grid_04, _, _, _, _ = make_occupancy_grid(args[1], 0.33, False)
    #grid_1, _, _, _, _ = make_occupancy_grid(args[1], 1, False)
    grid_1, _, _, _, _ = make_occupancy_grid(args[1], 0.7, False)

    my_fig,my_ax = create_map_png(grid_gates,minimum,maximum,x_c, y_c)

    ic = executionner(grid_gates,grid_04,grid_1,x_c, y_c, maximum, minimum, my_fig,my_ax)


    print("PATH EXECUTION --> running...")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
