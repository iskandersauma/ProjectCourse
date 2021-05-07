#!/usr/bin/env python
from __future__ import print_function

import rospy
import tf2_geometry_msgs
import tf2_ros
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from crazyflie_driver.msg import Position
import random
import math
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt


class Node_opt():
    def __init__(self,x = 0,y = 0,yaw = 0,parent = None):
		self.x = x
		self.y = y
		self.yaw = yaw
		self.parent = parent

class Node():
	def __init__(self,x,y,yaw = 0,heuristic = 0, parent = None, steps = 0):
		self.x = x
		self.y = y
		self.yaw = yaw
		self.parent = parent
		self.heuristic = heuristic
		self.steps = steps

class A_star():

	def __init__(self, start, finish, grid, maximum, minimum, x_c, y_c):
		self.start = Node(start[0][0],start[0][1],start[0][2],999999)
		self.end = Node(finish[0][0], finish[0][1],finish[0][2])
		self.node_list = [self.start]
		self.grid = grid
		self.maximum = maximum
		self.minimum = minimum
		self.x_c = x_c
		self.y_c = y_c

		self._start_time = rospy.Time.now()



	def collision(self,x,y):
		if(x >= self.maximum[0] - 0.1 or x <= self.minimum[0] + 0.1 or y >= self.maximum[1] - 0.1 or y <= self.minimum[1] + 0.1):
			return True


		x_est = int(x*100) + self.x_c
		y_est = int(y*100) + self.y_c

		if(self.grid[x_est][y_est] == ['occupied']):
			return True

		return False


	def check_grid_around_point(self,x,y):

		#Checking all points around in a given radius

		if x < len(self.grid) and y < len(self.grid[0]) and x >= 0 and y>= 0:
			if(self.grid[x ][y] == ['occupied']):
				return True
		return False


	def collision_on_line(self,x_0,y_0,x_1,y_1):

		x_0 = float(x_0*100) + float(self.x_c)
		y_0 = float(y_0*100) + float(self.y_c)
		x_1 = float(x_1*100) + float(self.x_c)
		y_1 = float(y_1*100) + float(self.y_c)

		if int(x_0-x_1) == 0 and int(y_0-y_1) == 0:
			return False

		#Computing linear function between the two points
		if abs(x_0-x_1) > abs(y_0-y_1):
			m = (y_0-y_1)/(x_0-x_1)
			p = y_0 - m*x_0
			for x in range(int(x_0),int(x_1),int(np.sign(x_1-x_0))):
				y = int(m*x+p)

				if self.check_grid_around_point(x,y):
					return True
			return False

		else:
			m = (x_0-x_1)/(y_0-y_1)
			p = x_0 - m*y_0
			for y in range(int(y_0),int(y_1),int(np.sign(y_1-y_0))):
				x = int(m*y+p)
				if self.check_grid_around_point(x,y):
					return True
			return False


	def lowest_heuristic(self):
		dList= []
		for i in range(len(self.node_list)):
			heuristic = self.node_list[i].heuristic
			dList.append(heuristic)
		index = dList.index(min(dList))
		return index

	def at_goal(self, x_1, y_1):
		D = 1
		dx = abs(x_1 - self.end.x)
		dy = abs(y_1 - self.end.y)
		d = D*(dx**2 + dy**2)**0.5
		if d < 0.1:
			return True
		else:
			return False

	def calculate_heuristic(self,x_1,y_1,yaw):
		D = 1
		dx1 = abs(x_1 - self.end.x)
		dy1 = abs(y_1 - self.end.y)
		imag_x = x_1 + 0.5*math.cos(yaw)
		imag_y = y_1 + 0.5*math.sin(yaw)

		d = D*(dx1**2 + dy1**2)**0.5

		if self.collision_on_line(x_1,y_1,imag_x,imag_y):
			return 3*d
		else:
			return d

	def step_function(self,x_1,y_1,yaw_1):
		x_1 = x_1 + 0.05*math.cos(yaw_1)
		y_1 = y_1 + 0.05*math.sin(yaw_1)
		return x_1, y_1

	def build_tree(self):

		while len(self.node_list):






			index = self.lowest_heuristic()
			closest_node = self.node_list[index]
			self.node_list.pop(index)



			current_time = rospy.Time.now()
			if current_time.secs - self._start_time.secs > 20:
				rospy.logerr("Path planning is taking too much time ! Returning computation done even if not finished...")
				return closest_node.parent


			crash = False
			x_0 = closest_node.x
			y_0 = closest_node.y
			yaw_list = [-5*math.pi/6, -3*math.pi/4,2*math.pi/3, -math.pi/2, -math.pi/3, -math.pi/4, -math.pi/6, 0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, 2*math.pi/3, 3*math.pi/4, 5*math.pi/6, math.pi]

			for i in range(len(yaw_list)):
				yaw_1 = yaw_list[i]
				counter = 0
				crash = False
				# Init to keep this node in memory !!
				x_1 = x_0
				y_1 = y_0
				while crash == False and counter < 5:
					x_1, y_1 = self.step_function(x_1, y_1,yaw_1)
					yaw_deg = math.degrees(yaw_1)

					if(self.collision(x_1,y_1)):
						crash = True

					if (self.at_goal(x_1, y_1) and crash == False):
						heuristic = self.calculate_heuristic(x_1,y_1,yaw_1)*closest_node.steps**0.4 #Prefer small distance first
						if(closest_node.parent == None):
							new_node  = Node(x_1, y_1, yaw_deg, heuristic,[(closest_node.x,closest_node.y,closest_node.yaw)] + [(x_1, y_1, yaw_1)] + [(self.end.x,self.end.y,self.end.yaw)], closest_node.steps + 1)
						else:
							new_node  = Node(x_1, y_1, yaw_deg, heuristic, closest_node.parent + [(closest_node.x,closest_node.y,closest_node.yaw)]+[(x_1 , y_1, yaw_deg)]+ [(self.end.x,self.end.y,self.end.yaw)],closest_node.steps + 1)
						self.node_list.append(new_node)
						return new_node.parent
					counter += 1

				'''if self.collision_on_line(x_0,y_0,x_1,y_1):
					crash = True'''

				if (counter > 0 and crash == False):
					heuristic = self.calculate_heuristic(x_1,y_1,yaw_1)*closest_node.steps**0.4
					if(closest_node.parent == None):
						new_node  = Node(x_1, y_1,yaw_deg, heuristic,[(closest_node.x,closest_node.y,closest_node.yaw)],closest_node.steps + 1)
					else:
						new_node  = Node(x_1, y_1,yaw_deg, heuristic, closest_node.parent + [(closest_node.x,closest_node.y,closest_node.yaw)],closest_node.steps + 1)
					self.node_list.append(new_node)



	def calculate_angle(self,x_1,y_1,x_new,y_new):
		angle = math.atan2(y_new - y_1,x_new - x_1)
		return angle

	def optimize_controls_bis(self,controls):

		if controls is None:
			return None

		opt_control = [(controls[0][0],controls[0][1],controls[0][2])]

		current_index = 0

		while current_index < len(controls)-1:

			try_index = len(controls) - 1

			x_i = controls[current_index][0]
			y_i = controls[current_index][1]

			x_r = controls[try_index][0]
			y_r = controls[try_index][1]

			if self.collision_on_line(x_i,y_i,x_r,y_r) and try_index == current_index +1:
				rospy.logwarn("Collision detected between two close points... shit might happen")

			while self.collision_on_line(x_i,y_i,x_r,y_r) and try_index > current_index :
				try_index = try_index - 1
				x_r = controls[try_index][0]
				y_r = controls[try_index][1]
			if current_index != try_index :
				current_index = try_index
			else :
				current_index = try_index + 1
			angle = math.degrees(self.calculate_angle(x_i,y_i,x_r,y_r))
			opt_control = opt_control + [(controls[current_index][0],controls[current_index][1],angle)]


		return opt_control


def check_obstacles_avoidance(start,finish,x_c,y_c,grid_list):

		N = len(grid_list)

		if N == 1:
			return 0

		collision_checks = np.zeros((N,1))

		x_0,y_0 = start[0][0],start[0][1]
		x_1,y_1 = finish[0][0], finish[0][1]

		x_0 = float(x_0*100) + float(x_c)
		y_0 = float(y_0*100) + float(y_c)
		x_1 = float(x_1*100) + float(x_c)
		y_1 = float(y_1*100) + float(y_c)

		if int(x_0-x_1) == 0 and int(y_0-y_1) == 0:
			return False

		#Computing linear function between the two points
		if abs(x_0-x_1) > abs(y_0-y_1):
			m = (y_0-y_1)/(x_0-x_1)
			p = y_0 - m*x_0
			for x in range(int(x_0),int(x_1),int(np.sign(x_1-x_0))):
				y = int(m*x+p)
				for grid_idx in range(N):
					if grid_list[grid_idx][x][y] == ['occupied']:
						collision_checks[grid_idx,0] = collision_checks[grid_idx,0] + 1
		else:
			m = (x_0-x_1)/(y_0-y_1)
			p = x_0 - m*y_0
			for y in range(int(y_0),int(y_1),int(np.sign(y_1-y_0))):
				x = int(m*y+p)
				for grid_idx in range(N):
					if grid_list[grid_idx][x][y] == ['occupied']:
						collision_checks[grid_idx,0] = collision_checks[grid_idx,0] + 1

		#Returning the map with fewer collision
		return np.argmin(collision_checks)


def plan_rout(start, finish, grid_list, x_c, y_c, maximum, minimum):

	z_choices = [0.33,0.7]
	idx =  check_obstacles_avoidance(start,finish,x_c,y_c,grid_list)
	z = z_choices[idx]
	grid = grid_list[idx]

	rospy.loginfo("Planning at altitude z = %s",z)

	tree = A_star(start,finish,grid, maximum, minimum,x_c, y_c)
	controls = tree.build_tree()
	print("no a* problems")
	optimal_controls = tree.optimize_controls_bis(controls)
	if optimal_controls is None:
		return None, None
	else:
		return optimal_controls + finish, z
