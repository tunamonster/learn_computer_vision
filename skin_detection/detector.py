# tutorial here http://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
import imutils
import numpy as np 
import argparse
import cv2

# construct arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to optional video")
args = vars(ap.parse_args())

# define upper and lower boundaries of HSV pixel intensities
# to be considered skin
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20,255,255], dtype = "uint8")

# load the video from the path
if args.get("video", True):
	camera = cv2.VideoCapture(args["video"])
# if no path is specified, open the webcam
else:
	camera = cv2.VideoCapture(0)

# loop over the frames in the video
while True:
	# get current frame
	(grabbed, frame) = camera.read()

	# if we are playing a video input and grabbed does not return
	# a frame, we have reached the end of the video.
	if args.get("video") and not grabbed:
		break

	# resize he frame and convert it to HSV color space, and
	# determine the HSV pixel intensities that fall into 
	# the specified upper and lower boundaries
	frame = imutils.resize(frame, width= 400)
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)

	# apply erosions and dilations 
	# on the mask using an elliptical kernel 
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))	
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	# blur the mask to help remove noise, then apply
	# the mask to the frame
	skinMask = cv2.GaussianBlur(skinMask, (3,3), 0)
	skin = cv2.bitwise_and(frame, frame, mask = skinMask)

	# show the skin in the image along with the mask 
	cv2.imshow("images", np.hstack([frame, skin]))

	# stop the loop if 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# close the camera
camera.release()	
cv2.destroyAllWindows()















