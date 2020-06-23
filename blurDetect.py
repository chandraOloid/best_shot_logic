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
import time
from pyimagesearch.blur_detector import detect_blur_fft


from os import listdir
from os.path import isfile, join
mypath = 'blurred_images'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)

for myfile in onlyfiles:
    image = cv2.imread(mypath + '/' +myfile)
    #print(image)
    #print(mypath + '/' + myfile)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (mean, blurry) = detect_blur_fft(gray, size=60, thresh=10, vis=-1 > 0)
    print(mean)
