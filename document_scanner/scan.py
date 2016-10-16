from transform import four_point_transform
import imutils
from skimage.filters import threshold_adaptive
import numpy as np 
import argparse 
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, 
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of old height
# to new height, clone, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert image to grayscale, blur, and find edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged =cv2.Canny(gray, 75, 200)

# show the original image and edge detected version
print("STEP 1: Edge Detection")
# cv2.imshow("edged", edged)
# cv2.imshow("original", image)
# cv2.waitKey(0)

# find the contours in the edged image
# keep only the largest, and initialize 
# screen contour

(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

print("length", len(cnts))
# loop over the contours
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if the approximated contour has four points, assume
	# the screen is found	
	if len(approx) == 4:
		screenCnt = approx
		break

# show the contour outline of the paper
# print("STEP 2: Find contours of paper")
# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()	

print("STEP 3: Resize and crop along the contours")
# apply the four point transform to obtain a top-down
# view of the original image

# break
# import code; code.interact(local=dict(globals(), **locals()))

warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)

# # convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 251, offset = 10)
warped = warped.astype("uint8") * 255
 
# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 600))
cv2.imshow("Scanned", imutils.resize(warped, height = 600))
cv2.waitKey(0)