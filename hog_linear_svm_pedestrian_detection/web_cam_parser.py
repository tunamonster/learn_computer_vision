# adapted http://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/ 
# to be used on a video stream. 
# code to handle video is from http://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np 
import argparse 
import imutils 
import cv2

# construct arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to optional video")
args = vars(ap.parse_args())

# load the video from the path
if args.get("video", True):
	camera = cv2.VideoCapture(args["video"])
# open the webcam if no path was specified
else:
	camera = cv2.VideoCapture(0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over video frames while they are streaming
while True:
	# get the current frame
	(grabbed, frame) = camera.read()

	# if we are playing a video input and grabbed does not return
	# a frame, we have reached the end of the video.
	if args.get("video") and not grabbed:
		break

	# resize the frame to increase processing speed
	frame = imutils.resize(frame, width = min(600, frame.shape[1]))
	# perform HSG 
	(rects, weights) = hog.detectMultiScale(frame, winStride = (5,5),
	padding = (4,4), scale = 1.05)

	# perform NMS on the detected areas
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=1)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# print results to frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


