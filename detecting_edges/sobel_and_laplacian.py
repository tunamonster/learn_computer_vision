# tutorial on page 11-26: http://www.pyimagesearch.com/static/ppao-sample-chapter.pdf
import numpy as np 
import argparse
import cv2

# set up the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image")
args = vars(ap.parse_args())

# convert the image to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# generate the gradient magnitude using the laplacian method, 
# represent the derivatives 
lap = cv2.Laplacian(gray, cv2.CV_64F)

lap = np.uint8(np.absolute(lap))


# Sobel gradient representation 
sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

cv2.imshow("original", image)
cv2.imshow("sobelX", sobelX)
cv2.imshow("sobelY", sobelY)
cv2.imshow("combined sobel ops", sobelCombined)
cv2.waitKey(0)

# pause the program and enable code interaction
# import code; code.interact(local=dict(globals(), **locals()))
