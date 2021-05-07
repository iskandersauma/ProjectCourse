#!/usr/bin/env python
from __future__ import print_function

import math
import rospy
import cv2
import sys
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#from scipy.ndimage import label, generate_binary_structure
from keras.preprocessing import image
from keras.models import load_model
from pras_project.msg import CroppedImageObject
from pras_project.msg import ObjectPose

class object_analyser:

	def __init__(self,_c1,_p1,_c2,path):

		self.bridge = CvBridge()
		#INPUT
		self.image_sub_warped_rect = rospy.Subscriber("/vision/imageCropped", CroppedImageObject, self.callback_ImageCropped)
		#OUTPUT
		self.object_pub = rospy.Publisher("/vision/object_pose", ObjectPose, queue_size=2)
		self.cercleB_pub_reoriented = rospy.Publisher("/vision/debug/cercle_oriented/blue", Image, queue_size=2)
		self.cercleR1_pub_reoriented = rospy.Publisher("/vision/debug/cercle_oriented/red1", Image, queue_size=2)
		self.cercleR2_pub_reoriented = rospy.Publisher("/vision/debug/cercle_oriented/red2", Image, queue_size=2)
		self.cercleRB_pub_reoriented = rospy.Publisher("/vision/debug/cercle_oriented/redblue", Image, queue_size=2)


		self.mask_inner_circle = _c1
		self.mask_big_circle = _c2
		self.points_outer_circle = _p1
		self.pC2_x = self.points_outer_circle[:,0]
		self.pC2_y = self.points_outer_circle[:,1]

		self.rectangles_classes = ["airport","residential","NOPE"]
		self.triangles_classes = ["dangerous_curve_left","dangerous_curve_right","junction","road_narrows_from_left","road_narrows_from_right","circulation_warning","NOPE"]
		self.circles_classes = ["follow_left","follow_right","no_bicycle","no_heavy_truck","stop","no_parking","no_stopping_and_parking","NOPE"]

		self.circles_1_classes = ["follow_left","follow_right","NOPE"]
		self.circles_2_classes = ["no_bicycle","no_heavy_truck","stop","NOPE"]
		self.circles_3_classes = ["no_parking","no_stopping_and_parking","NOPE"]

		self.path = path
		self.rectangle_classifier = load_model(self.path + "Rectangles.h5")
		self.rectangle_classifier._make_predict_function()

		self.triangle_classifier = load_model(self.path + "Triangles.h5")
		self.triangle_classifier._make_predict_function()

		#self.circle_classifier = load_model(self.path + "Cercles.h5")
		#self.circle_classifier._make_predict_function()

		self.circle_1_classifier = load_model(self.path + "Cercles_1.h5")
		self.circle_1_classifier._make_predict_function()

		self.circle_2_classifier = load_model(self.path + "Cercles_2.h5")
		self.circle_2_classifier._make_predict_function()

		self.circle_3_classifier = load_model(self.path + "Cercles_3.h5")
		self.circle_3_classifier._make_predict_function()

	def callback_ImageCropped(self,imageCropped):

		try:
			cv_image = self.bridge.imgmsg_to_cv2(imageCropped.image, "bgr8")
		except CvBridgeError as e:
			print(e)
			return

		if imageCropped.form == 2:
			cv_image = cv2.resize(cv_image,(64,64))
			name,prob = self.refine_circle_object(cv_image)
			#cv_image = cv2.resize(cv_image,(32,32))
			#name,prob = self.get_name_with_CNN(imageCropped.form, cv_image)
			rospy.loginfo("p = %s ; %s",prob,name)
		else:
			cv_image = cv2.resize(cv_image,(32,32))
			#name,prob = self.get_name_with_CNN(imageCropped.form, cv_image)

			#cv_image = cv2.resize(cv_image,(32,32))
			name,prob = self.get_name_with_CNN(imageCropped.form, cv_image)

		rospy.loginfo_throttle(0.2,("p = %s ; %s",prob,name))

		if name != 'NOPE':
			self.publish_to_post_treatment(imageCropped.pose, name, imageCropped.confidence, prob)

		return



	def refine_circle_object(self,img):

		# Getting the color mask inside the circle
		mask_R, mask_B = self.get_mask_color(img)
		mask_R = mask_R & self.mask_big_circle
		mask_B = mask_B & self.mask_big_circle

		blue_proportion = np.sum(mask_B) / (np.sum(mask_R) + np.sum(mask_B))
		red_proportion = np.sum(mask_R) / (np.sum(mask_R) + np.sum(mask_B))

		#############
		# PASS HERE : follow_left, follow_right
		#############
		if blue_proportion > 0.98:
			lower = np.array([40])
			upper = np.array([255])

			theta, img = self.find_angle_of_motif(img,lower, upper)
			if theta != 0:
				# Rotate the image
				M = cv2.getRotationMatrix2D((32,32),theta,1)
				img = cv2.warpAffine(img,M,(64,64))

			test_image = image.img_to_array(img)
			test_image = np.expand_dims(test_image, axis = 0)
			result = self.circle_1_classifier.predict(test_image)
			id_img = np.argmax(result)
			prob = result[0][id_img]
			name = self.circles_1_classes[id_img]

		else:
			#############
			# PASS HERE : no_bicycle, no_heavy_truck, stop
			#############
			if red_proportion > 0.98:

				lower = np.array([100])
				upper = np.array([255])

				theta, img = self.find_angle_of_motif(img,lower, upper)
				if theta != 0:
					# Rotate the image
					M = cv2.getRotationMatrix2D((32,32),theta,1)
					img = cv2.warpAffine(img,M,(64,64))

				test_image = image.img_to_array(img)
				test_image = np.expand_dims(test_image, axis = 0)
				result = self.circle_2_classifier.predict(test_image)
				id_img = np.argmax(result)
				prob = result[0][id_img]
				name = self.circles_2_classes[id_img]

			#############
			# PASS HERE : no_parking, no_parking_and_stoping
			#############
			else:
				lower = np.array([0])
				upper = np.array([20])

				theta,points = self.find_points_of_RED_motif(img,lower, upper)

				name = self.mlParams(points,theta)
				prob = 0.5

		return name,prob

	def mlParams(self,X,theta):
		Npts,Ndims = np.shape(X)
		X[:,0] = X[:,0]*np.cos(-theta)
		X[:,1] = X[:,1]*np.sin(-theta)

		mu = np.zeros((1,Ndims))
		sigma = np.zeros((Ndims,Ndims))

		for n in range(Npts):
			mu[0,:] += X[n,:]
		mu[0,:] /= Npts

		for n in range(Npts):
			sigma[ :, :] += np.diag((X[n, :]-mu[0, :])**2)
		sigma[:, :] /= Npts

		s11,s22 = sigma[0,0],sigma[1,1]

		if max([s11/s22,s22/s11]) > 1.5:
			name = 'no_parking'
		else:
			name = 'no_stopping_and_parking'

		return name



	def find_angle_of_motif(self,img,l_b,u_b):

		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#Get the mean color of the outer circle
		m = np.mean( img_gray[self.pC2_x,self.pC2_y] ,axis = 0)
		#Base the image on this color
		base_img = abs(img_gray - m)
		mask_black = cv2.inRange(base_img, l_b, u_b)
		#find the points of the same color inside the sign
		MASK = mask_black & self.mask_inner_circle
		points = np.argwhere(MASK != 0)

		if len(points) > 0:
			p_x = points[:,0]
			p_y = points[:,1]

			#img[p_x,p_y] = [0,255,0]
			# Fitting a line on those points
			A = np.vstack([p_y, np.ones(len(p_y))]).T
			a, _ = np.linalg.lstsq(A, p_x, rcond=None)[0]
			# Get the vector colinear to the line
			norm = np.sqrt(1+a*a)
			vx,vy = 1/norm,a/norm
			# Angle of rotation of the line
			theta = np.sign(vy)*abs(math.acos(vx))
			theta = theta * 180/np.pi

			return theta,img

		else:
			return 0, img


	def find_points_of_RED_motif(self,img,l_b,u_b):
		img_gray = img[:,:,2]
		#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#Get the mean color of the outer circle
		m = np.mean( img_gray[self.pC2_x,self.pC2_y] ,axis = 0)
		#Base the image on this color
		base_img = abs(img_gray - m)
		mask_black = cv2.inRange(base_img, l_b, u_b)
		#find the points of the same color inside the sign
		MASK = mask_black & self.mask_inner_circle
		points = np.argwhere(MASK != 0)

		if len(points) <10:
			u_b = np.array([40])
			mask_black = cv2.inRange(base_img, l_b, u_b)

			#find the points of the same color inside the sign
			MASK = mask_black & self.mask_inner_circle
			points = np.argwhere(MASK != 0)

		elif len(points) >100:
			u_b = np.array([10])
			mask_black = cv2.inRange(base_img, l_b, u_b)

			#find the points of the same color inside the sign
			MASK = mask_black & self.mask_inner_circle
			points = np.argwhere(MASK != 0)

		if len(points) > 0:
			p_x = points[:,0]
			p_y = points[:,1]

			#img[p_x,p_y] = [0,255,0]
			# Fitting a line on those points
			A = np.vstack([p_y, np.ones(len(p_y))]).T
			a, _ = np.linalg.lstsq(A, p_x, rcond=None)[0]
			# Get the vector colinear to the line
			norm = np.sqrt(1+a*a)
			vx,vy = 1/norm,a/norm
			# Angle of rotation of the line
			theta = np.sign(vy)*abs(math.acos(vx))
			theta = theta

			return theta,points

		else:
			return 0, points



	def get_mask_color(self,raw_img):

		# Convert BGR to HSV
		hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)

		# Red
		lower = np.array([170,35,51])
		upper = np.array([180,255,180])
		mask_red1 = cv2.inRange(hsv, lower, upper)
		lower = np.array([0,35,51])
		upper = np.array([30,255,180])
		mask_red2 = cv2.inRange(hsv, lower, upper)
		# Blue
		lower = np.array([95,70,70])
		upper = np.array([126,220,220])
		mask_blue = cv2.inRange(hsv, lower, upper)

		mask_red = mask_red1 + mask_red2

		#Return all the masks
		return mask_red, mask_blue



	def get_name_with_CNN(self,form,img):

		test_image = image.img_to_array(img)
		test_image = np.expand_dims(test_image, axis = 0)

		if form == 0:
			result = self.rectangle_classifier.predict(test_image)
			id_img = np.argmax(result)
			prob = result[0][id_img]
			name = self.rectangles_classes[id_img]

		if form == 1:
			result = self.triangle_classifier.predict(test_image)
			id_img = np.argmax(result)
			prob = result[0][id_img]
			name = self.triangles_classes[id_img]

		if form == 2:
			result = self.circle_classifier.predict(test_image)
			id_img = np.argmax(result)
			prob = result[0][id_img]
			name = self.circles_classes[id_img]

		return name, prob


	def publish_to_post_treatment(self, pose, name, conf, prob):

		my_object = ObjectPose()

		my_object.header = pose.header
		my_object.name = name
		my_object.pose = pose
		my_object.confidence = conf*prob

		self.object_pub.publish(my_object)

		return


def get_mask_circulaire(r):
	# location center
	xc, yc = 32, 32
	# x and y coordinates per every pixel of the image
	x, y = np.meshgrid(np.arange(64), np.arange(64))
	# squared distance from the center of the circle
	d2 = (x - xc)**2 + (y - yc)**2
	# mask is True inside of the circle
	mask = d2 < r**2

	return mask

def main(argv=sys.argv):
	rospy.init_node('object_classification')

	args = rospy.myargv(argv=argv)

	mask_F1 = get_mask_circulaire(28)
	mask_F2 = get_mask_circulaire(26)
	mask_F3 = get_mask_circulaire(22)
	res = np.zeros((64,64))
	np.logical_xor(mask_F1, mask_F2,out=res)
	zone = np.array(np.argwhere(res != 0))

	ic = object_analyser(mask_F3,zone, mask_F1,args[1])

	print("CLASSIFICATION --> running...")
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

	cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
