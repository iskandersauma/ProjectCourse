#!/usr/bin/env python
from __future__ import print_function

import math
import rospy
import cv2
import sys
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from pras_project.msg import CroppedImageObject

#YOU HAVE THE LAST VERSION

class image_analyser:

	def __init__(self,rect_PnP,tria_PnP,circ_PnP):

		# INPUT
		#self.image_sub = rospy.Subscriber("/cf1/camera/image_rect_color", Image, self.callback_get_image)
		self.image_sub = rospy.Subscriber("/camera/image_raw/compressed", CompressedImage, self.callback_get_image)
		# OUTPUT
		self.croppedImage_pub = rospy.Publisher("/vision/imageCropped", CroppedImageObject, queue_size=2)

		# INIT VARIABLES
		self.bridge = CvBridge()

		self.image_pub_debug_detect = rospy.Publisher("/vision/debug/detection", Image, queue_size=2)
		self.image_pub_debug_color = rospy.Publisher("/vision/debug/color", Image, queue_size=2)
		self.image_pub_debug_edges = rospy.Publisher("/vision/debug/edges", Image, queue_size=2)

		self.min_area_shape = 400 #The miniminu area (in pixel) of the shape to be considered as a potetial object
		self.max_image_retry = 2 #CAN BE VERY COMPUTATIONALY HEAVY !
		self.blurring_factor = 3 #Blurring during the shape detection
		self.min_matching_coef = 0.1 #The minimum coef of match betweew the shape and the color mask to be considered as an object

		self.precision_polyapprox = 0.01#The precision of the polynomial approximation in the first layer of detection
		self.refining_max_iter = 20 #The maximum number of iteration to refine the curve as the detected shape
		self.refining_increase_rate = 1.2 #Rate of increase epsilon in the curve approximation (1.2 => 20%)
		self.refining_decreasing_rate = 0.9 #Rate of decreasing epsilon in the curve approximation

		self.ratio_shape_rect = 1.5 #Ratio W/H for a rectangle sign
		self.ratio_shape_tri = 1.126 #Ratio W/H for a triangular sign
		self.ratio_shape_cir = 1 #Ratio W/H for a circular sign
		self.max_deformation = 2.5 #Maximum w/h or h/w tolerated

		self.camera_matrix = np.array([[231.250001,0,320.519378],
			[0,231.065552,240.631482],
			[0,0,1]], dtype = "float32")
		self.distortion_coef = np.array([0.061687, -0.049761, -0.008166, 0.004284], dtype = "float32")

		self._rect_PnP = rect_PnP
		self._tria_PnP = tria_PnP
		self._circ_PnP = circ_PnP


	# Main callback getting all images from ros
	def callback_get_image(self,image_message):

		# Convert the image from ROS to OPENCV format
		try:
			#cv_image = self.bridge.imgmsg_to_cv2(image_message, "bgr8")
			cv_image = self.bridge.compressed_imgmsg_to_cv2(image_message, "bgr8")
		except CvBridgeError as e:
			print(e)
			rospy.logerr("Error 1")
			return

		self.detect_object_candidates(cv_image,1)
		return



	# Find the 3D pose of a rectangle given the 2D pixel poses
	def get_position_PnP(self, form, img_p):

		if form == 0:
			obj_p = self._rect_PnP
		if form == 1:
			obj_p = self._tria_PnP
		if form == 2:
			obj_p = self._circ_PnP

		_, rvec, tvec = cv2.solvePnP(obj_p,img_p,self.camera_matrix,self.distortion_coef)

		return rvec,tvec


	# Publishing the output, image + pose
	def publish_imageCropped(self,IMAGE,FORM,POS,ANG,CONF):

		X, Y, Z = POS
		roll,pitch,yaw = ANG

		#Creating the image object for the image
		imageObject = CroppedImageObject()

		imageObject.header.stamp = rospy.Time.now()
		imageObject.header.frame_id = 'cf1/camera_link'
		imageObject.pose.header.stamp = rospy.Time.now()
		imageObject.pose.header.frame_id = 'cf1/camera_link'

		imageObject.form = FORM

		try:
			imageObject.image = self.bridge.cv2_to_imgmsg(IMAGE, "bgr8")
		except CvBridgeError as e:
			print(e)
			return

		imageObject.pose.pose.position.x = X
		imageObject.pose.pose.position.y = Y
		imageObject.pose.pose.position.z = Z
		(imageObject.pose.pose.orientation.x,
		imageObject.pose.pose.orientation.y,
		imageObject.pose.pose.orientation.z,
		imageObject.pose.pose.orientation.w) = quaternion_from_euler(math.radians(roll),
														math.radians(pitch),
														math.radians(yaw),axes='sxyz')

		imageObject.confidence = CONF

		self.croppedImage_pub.publish(imageObject)

		return


	# Main function
	def detect_object_candidates(self, cv_image, Nb_tries):

		#Getting the shape mask and the boundaries of each zone, as a rectangle
		shape_mask,obj_shapes,obj_zones = self.get_mask_object_contouring(cv_image)

		#Getting the color mask
		mask_R, mask_B, _, mask_BW = self.get_mask_object_color(cv_image)
		color_mask = mask_R + mask_B



		#################DEBUG
		#Publishing if we look at this screen
		connections = self.image_pub_debug_color.get_num_connections()
		res = cv2.bitwise_and(cv_image, cv_image, mask= color_mask)
		if connections > 0:
			try:
				self.image_pub_debug_color.publish(self.bridge.cv2_to_imgmsg(res, "bgr8"))
			except CvBridgeError as e:
				print(e)
				rospy.logerr("Error 2")
				return
		#####################
		


		#Shape AND color mask
		color_mask = (shape_mask & color_mask)
		#Shape AND black and white mask
		b_w_mask = (shape_mask & mask_BW)

		# Loop over all the candidates
		for i in range(len(obj_zones)):
			#Treating zone by zone
			zone = obj_zones[i]
			x,y,w,h = zone


			shape_proportion = np.sum(shape_mask[y:y+h, x:x+w])/255.
			color_proportion = np.sum(color_mask[y:y+h, x:x+w])/255.
			black_and_white =  np.sum(b_w_mask[y:y+h, x:x+w])/255.
			#Matching of the color inside the shape delimitation
			coef_matching = (color_proportion + black_and_white)/shape_proportion

			if coef_matching > self.min_matching_coef and color_proportion > black_and_white:
				#rospy.loginfo(coef_matching)

				#Method to define the form of the curve and get the image corrected perspective
				res_form, res_img, res_points = self.get_warp_image(obj_shapes[i], zone, cv_image)

				if res_form is not None:

					# We last check if the result seems possible before publishing

					if self.check_validity(res_form,zone,mask_R,mask_B):
						cv2.rectangle(cv_image,(zone[0],zone[1]),(zone[0]+zone[2],zone[1]+zone[3]),(0,0,255),2)
						cv2.drawContours(cv_image,[obj_shapes[i]], 0, (0,255,0),3 )

						Angles, Position = self.get_position_PnP(res_form, res_points)

						self.publish_imageCropped(res_img, res_form, Position, Angles, coef_matching)


					# We retry to do the shape detection if necessary
					elif (Nb_tries <= self.max_image_retry):
						rospy.logdebug("\n********** MISCLASSIFICATION !*************\n TRYING TO RUN AGAIN ! ")
						rospy.logdebug("Current guess was : %s",res_form)
						rospy.logdebug("Number of tries done yet : %s",Nb_tries)
						rospy.logdebug("***********************\n\n")

						restricted_img = cv_image[zone[1]:zone[1]+zone[3], zone[0]:zone[0]+zone[2]]
						self.detect_object_candidates(restricted_img, Nb_tries+1)



		#################DEBUG
		#Publishing if we look at this screen
		connections = self.image_pub_debug_detect.get_num_connections()
		if connections > 0:
			try:
				self.image_pub_debug_detect.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
			except CvBridgeError as e:
				print(e)
				return
		#####################


	# Last check of a candidate given its form
	def check_validity(self,shape_type,zone,mask_R,mask_B):
		x,y,w,h = zone

		#If rectangle, we check the color blue
		if shape_type == 0:
			R_proportion = np.sum(mask_R[y:y+h, x:x+w])/255
			B_proportion = np.sum(mask_B[y:y+h, x:x+w])/255
			#Rectangles have no red
			if R_proportion > B_proportion:
				return False

		#If rectangle, we check the color blue
		if shape_type == 1:
			R_proportion = np.sum(mask_R[y:y+h, x:x+w])/255
			B_proportion = np.sum(mask_B[y:y+h, x:x+w])/255
			#Triangles have no blue
			if B_proportion > R_proportion:
				return False

		return True



	### Creating the four border points of a triangle considering the perspective and ordering them clockwise starting from top-left
	def order_points_tri(self,curve):

		#Rebuild an array form polynomial curve
		pts =  np.array([curve[:,0]], dtype = "float32")[0]
		tri = np.zeros((3, 2), dtype = "float32")
		points = np.zeros((4, 2), dtype = "float32")

		#top point will be the one with smallest y
		top_idx = np.argmin(pts[:,1])
		tri[1] = pts[top_idx]

		#bottom right will be the one with min x if differs from the top points
		if np.argmin(pts[:,0]) != top_idx:
			bl_idx = np.argmin(pts[:,0])
			#br_idx = np.argmax(s)
			tri[0] = pts[bl_idx]
			#Only left the last index
			br_idx = 3 - top_idx - bl_idx
			tri[2] = pts[br_idx]
		else:
			br_idx = np.argmax(pts[:,0])
			#br_idx = np.argmax(s)
			tri[2] = pts[br_idx]
			#Only left the last index
			bl_idx = 3 - top_idx - br_idx
			tri[0] = pts[bl_idx]

		d_x = int((tri[2][0]-tri[0][0])/2.0)
		d_y = int((tri[2][1]-tri[0][1])/2.0)

		points[0] = [tri[1][0] - d_x,tri[1][1] - d_y]
		points[1] = [tri[1][0] + d_x,tri[1][1] + d_y]
		points[3] = tri[0]
		points[2] = tri[2]

		return points


	# Ordering the corners points of a rectangle clockwise starting from top left
	def order_points_rect(self,pts):

		rect = np.zeros((4, 2), dtype = "float32")

		# the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
		s = pts.sum(axis = 1)
		rect[0] = pts[np.argmin(s)]
		rect[2] = pts[np.argmax(s)]

		# the top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
		diff = np.diff(pts, axis = 1)
		rect[1] = pts[np.argmin(diff)]
		rect[3] = pts[np.argmax(diff)]

		return rect


	# Calculate the width and height of a rectangle
	def get_W_H(self,R):

		(tl, tr, br, bl) = R
		# maximum distance between bottom-right and bottom-left x-coordiates or the top-right and top-left x-coordinates
		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		W = max(int(widthA), int(widthB))

		# maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		H = max(int(heightA), int(heightB))

		return W, H


	# Estimate the form of a curve with estimators between rectangle triangle and circle
	def estimate_form(self,curve):

		area = cv2.contourArea(curve)
		peri = cv2.arcLength(curve,True)

		pts =  np.array([curve[:,0]], dtype = "float32")[0]

		estim_height = np.max(pts[:,1]) - np.min(pts[:,1])
		estim_width = np.max(pts[:,0]) - np.min(pts[:,0])

		#Estimation of Area/Perimeter
		estim_theory = area/peri

		estim1_rectangle = estim_height*(0.5 - estim_height/peri)
		estim2_rectangle = estim_width*(0.5 - estim_width/peri)

		estim1_triangle = estim_height/6
		estim2_triangle = peri*np.sqrt(3)/36

		estim_circle = peri/(4*np.pi)

		#computing relative error
		relat_err_rec1 = abs(estim_theory - estim1_rectangle)/estim_theory
		relat_err_rec2 = abs(estim_theory - estim2_rectangle)/estim_theory

		relat_err_tri1 = abs(estim_theory - estim1_triangle)/estim_theory
		relat_err_tri2 = abs(estim_theory - estim2_triangle)/estim_theory

		relat_err_cir = abs(estim_theory - estim_circle)/estim_theory

		err_vec = [np.mean([relat_err_rec1,relat_err_rec2]), np.mean([relat_err_tri1,relat_err_tri2]), relat_err_cir]

		#the shape is a circe : return the result
		if np.argmin(err_vec) == 2:
			return 2

		#Making the difference between a triangle and a rectangle
		else:

			# Try to fit a bounding box
			rect = cv2.minAreaRect(curve)
			box = cv2.boxPoints(rect)
			area_rect = cv2.contourArea(box)

			# Try to fit a triangle
			epsilon = 0.01*peri
			refined_Curve_tri = curve
			k = 1
			while len(refined_Curve_tri) != 3 and (k < self.refining_max_iter):
				if len(refined_Curve_tri) > 3:
					epsilon = self.refining_increase_rate*epsilon
				else:
					epsilon = self.refining_decreasing_rate
					#rospy.logerr("Error on fitting triangle : too few sides")

				refined_Curve_tri = cv2.approxPolyDP(curve,epsilon,True)
				k=k+1

			if len(refined_Curve_tri) != 3:
				rospy.logdebug("Strange shape cannot fit any triangle : assuming to be a rectangle")
				return 0

			area_tri = cv2.contourArea(refined_Curve_tri)

			#Error of fitting
			err_rect = abs(area_rect - area)/area
			err_tri = abs(area_tri - area)/area

			return np.argmin([err_rect,err_tri])



	# Lowering the number of sides of a curve to fit to the input
	def refine_curve(self,curve,nb_sides):
		#REFINE the curve to get the proper shape
		perimeter = cv2.arcLength(curve,True)
		epsilon = 0.01*perimeter
		refined_Curve = curve
		k = 1

		while len(refined_Curve) != nb_sides and (k < self.refining_max_iter):
			if len(refined_Curve) > nb_sides:
				epsilon = self.refining_increase_rate*epsilon
			else:
				epsilon = self.refining_decreasing_rate*epsilon

			refined_Curve = cv2.approxPolyDP(curve,epsilon,True)
			k=k+1

		#Did not managed to fit the disired curve
		if len(refined_Curve) != nb_sides:

			rospy.logdebug("*****************************************************************************")
			rospy.logdebug("Strange shape : gave up on it !")
			rospy.logdebug("I saw a form that is supposed to have this number os sides :")
			rospy.logdebug(nb_sides)
			rospy.logdebug("But could only refine until sides = ")
			rospy.logdebug(len(refined_Curve))
			rospy.logdebug("*****************************************************************************")
			return None

		#Return the refined curve
		else:
			return refined_Curve


	# Output frame in pixel for the perspective transform
	def output_frame(self,ratio,w_src,h_src):

		if w_src < h_src:
			width = int(w_src)
			height = int(width/ratio)
		else:
			height = int(h_src)
			width = int(height*ratio)

		#Build the destination shape
		points = np.array([
			[0,0],
			[width-1,0],
			[width-1,height-1],
			[0,height-1]], dtype = "float32")

		return points,width,height


	# Function that gves the maximum ration of w/h or h/w
	def get_max_deformation(self,w,h):

		if w == 0 or h == 0:
			return self.max_deformation + 1
		else:
			return max([float(w)/float(h), float(h)/float(w)])



	def get_warp_image(self,curve_in,zone,img):

		# [RECTANGLE, TRIANGLE, CIRCLE]
		sides = [4,3,None]#Number of sides of respective form
		form = self.estimate_form(curve_in)

		### RECTANGLE ###
		if form == 0:
			#Refine the curve
			curve = self.refine_curve(curve_in,sides[form])
			if curve is not None:
				#Sort the points
				pts =  np.array([curve[:,0]], dtype = "float32")[0]#Rebuild an array form polynomial curve
				points_In = self.order_points_rect(pts)

				w,h = self.get_W_H(points_In)
				coed_def = self.get_max_deformation(w,h)

				if coed_def < self.max_deformation:
					points_Out,W,H = self.output_frame(self.ratio_shape_rect,w,h)#Build the destination shape

					M = cv2.getPerspectiveTransform ( points_In, points_Out )
					img_warped = cv2.warpPerspective (img, M, (W,H),cv2.INTER_NEAREST )

					#Return true to draw the contour
					return form, img_warped, points_In

		### TRIANGLE ###
		elif form == 1:
			#Refine the curve
			curve = self.refine_curve(curve_in,sides[form])
			if curve is not None:
				#Sort the points
				points_In = self.order_points_tri(curve)

				w,h = self.get_W_H(points_In)
				coed_def = self.get_max_deformation(w,h)

				if coed_def < self.max_deformation:
					points_Out,W,H = self.output_frame(self.ratio_shape_tri,w,h)#Build the destination shape

					M = cv2.getPerspectiveTransform ( points_In, points_Out )
					img_warped = cv2.warpPerspective (img, M, (W,H),cv2.INTER_NEAREST )

					#Return true to draw the contour
					return form, img_warped, points_In

		### CIRCLE ###
		elif form == 2:
			ellip = cv2.fitEllipse(curve_in)
			box_ellipse = cv2.boxPoints(ellip)

			#Sort the points
			points_In = self.order_points_rect(box_ellipse)

			w,h = self.get_W_H(points_In)
			coed_def = self.get_max_deformation(w,h)

			if coed_def < self.max_deformation:
				points_Out,W,H = self.output_frame(self.ratio_shape_cir,w,h)#Build the destination shape

				M = cv2.getPerspectiveTransform ( points_In, points_Out )
				img_warped = cv2.warpPerspective (img, M, (W,H),cv2.INTER_NEAREST )

				#Return true to draw the contour
				return form, img_warped, points_In

		return None, None, None



	def get_mask_object_color(self,raw_img):
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
		# Yellow VERY BAD MASK
		lower = np.array([50,50,50])
		upper = np.array([90,160,240])
		mask_yellow = cv2.inRange(hsv, lower, upper)

		#Black and White
		lower = np.array([0,0,0])
		upper = np.array([180,25,45])
		mask_black = cv2.inRange(hsv, lower, upper)

		mask_red = mask_red1 + mask_red2

		#Return all the masks
		return mask_red, mask_blue, mask_yellow, mask_black



	def get_mask_object_contouring(self,raw_img):

		### EDGE DETECTION ###
		#Convert image to gray and blur it
		gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

		#Equalize histogram, adjust brightness and improve contrast
		norm_img = cv2.equalizeHist(gray_img)

		#blurring the image
		blur_img = cv2.blur(norm_img,(self.blurring_factor,self.blurring_factor))
		#Detecting the edges
		edges = cv2.Canny(blur_img,10,200)
		#Finfing the contours
		_, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


		#################DEBUG
		#Publishing if we look at this screen
		connections = self.image_pub_debug_edges.get_num_connections()
		if connections > 0:
			try:
				self.image_pub_debug_edges.publish(self.bridge.cv2_to_imgmsg(edges, "8UC1"))
			except CvBridgeError as e:
				print(e)
				rospy.logerr("Error 2")
				return
		#####################


		object_candidates = []
		object_zones = []
		#Analyze contours one by one
		for cnt in contours:

			perimeter = cv2.arcLength(cnt,True)
			epsilon = self.precision_polyapprox*perimeter
			approxCurve = cv2.approxPolyDP(cnt,epsilon,True)

			#Checking if the contour is convex
			if cv2.isContourConvex(approxCurve):

				#Filtering the very small contours in pixel dimension
				area = cv2.contourArea(approxCurve)
				if area > self.min_area_shape:
					object_candidates.append(approxCurve)
					object_zones.append(cv2.boundingRect(approxCurve))

		mask_shape = np.zeros(raw_img.shape[:2], np.uint8)
		#Draw all the contours on the image and fill in to get a mask
		cv2.drawContours(mask_shape, object_candidates, -1, (255),-1)

		return mask_shape, object_candidates, object_zones



# Create 3D points rectangles for future PnP solver
def PnP_solver_ObjectPoints(ratio):

	height = 0.2 #20 cm
	width = height*ratio
	#Build the destination shape
	points = np.array([
		[0,0,0],
		[width,0,0],
		[width,height,0],
		[0,height,0]], dtype = "float32")

	return points



def main(args):
	rospy.init_node('object_detection')

	rect_PnP = PnP_solver_ObjectPoints(1.5)
	tria_PnP = PnP_solver_ObjectPoints(1.126)
	circ_PnP = PnP_solver_ObjectPoints(1)

	ic = image_analyser(rect_PnP, tria_PnP, circ_PnP)

	print("DETETCTION --> running...")
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

	cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
