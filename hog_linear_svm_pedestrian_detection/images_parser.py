# tutorial here http://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np 
import argparse 
import imutils 
import cv2

# construct the argument parsers
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True, help = "path to images folder")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
for imagePath in paths.list_images(args["images"]):
	# load and resize the image to reduce detection time
	# and improve detection accuracy
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width = min(400, image.shape[1]))
	orig = image.copy()

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride = (4,4),
		padding = (8,8), scale = 1.05)

	# draw the bounding boxes
	for (x, y, w, h) in rects: 
		cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 0, 255), 2)

	# apply non-maxima suppresion to the bounding boxes using
	# a large overlap treshold to maintain people standing 
	# close to each other 
	rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs = None, overlapThresh = 0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show information on the number of bounding boxes
	filename = imagePath[imagePath.rfind("/") + 1:]
	print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))

cv2.imshow("before NMS", orig)
cv2.imshow("After NMS", image)
cv2.waitKey(0)



