# USAGE
# python detect_face_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

EAR_THRESH = 0.23
VARIANCE_THRESH = 64

def eye_aspect_ratio(eye):
	#print(eye)
	
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	print('ear = ', ear)
	# return the eye aspect ratio
	return ear

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

# construct the argument parser and parse the arguments
"""
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print('faces = ', faces)

for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = image[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_gray)
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	cv2.imshow('img',image)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
"""

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_cascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_eye.xml')
left_eye_cascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_righteye_2splits.xml')
#nose_cascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_mcs_nose2.xml')
#mouth_cascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_mcs_mouth.xml')

# load the input image, resize it, and convert it to grayscale
#cap = cv2.VideoCapture(0)
#image = cv2.imread('images/1.jpg')
image = cv2.imread('images/8.jpeg')
image = cv2.imread('images/9.jpeg')

# ksize 
ksize = (10, 10) 
# Using cv2.blur() method  

if 0: 
	image = cv2.blur(image, ksize)  

def bestShotImg(image):
	#image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
	#print(nose_rects)
	variance = variance_of_laplacian(image)
	#print(variance)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	"""
	LeftEyeVisible = 0 or 1 ( 0 not visible ) 
	RightEyeVisible = 0 or 1 
	LeftEyeOpen = 0 or 1
	RightEyeOpen = 0 or 1
	NoseVisible = 0 or 1
	MouthVisible = 0 or 1 
	ImageBlur = 0 ---- 1 scale ( 0.1, 0.3 etc. 0 for no blur, 1 for blurred image ) 
	LandmarkScore = ( add all 0 and 1 of facial Landmark ) 
	"""
	res = {'leye': 0, 'reye': 0, 'leyeopen': 0, 'reyeopen':0, 'nosevisible': 1, 'mouthvisible':1, 'imgBlur': 0, 'landmarkScore': 0}
	if variance < VARIANCE_THRESH: 
		res['imgBlur'] = 1
	else: 
		res['imgBlur'] = 0
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)

			#print('<---------------------->')
			res['leye'] = 1
			res['reye'] = 1
			if name == 'left_eye':
				ear = eye_aspect_ratio(shape[i:j+1])
				if ear > EAR_THRESH:
					res['leyeopen'] = 1
					
			if name == 'right_eye':		
				ear = eye_aspect_ratio(shape[i:j+1])
				if ear > EAR_THRESH:
					res['reyeopen'] = 1
					
			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
			
			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y:y + h, x:x + w]
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
			cv2.imshow('img', image)
			#key = cv2.waitKey(0)
	return res

res = bestShotImg(image)
print(res)