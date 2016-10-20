# tutorial here http://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
from skimage.exposure import rescale_intensity
import numpy as np 
import argparse
import cv2
import imutils

def convolve(image, kernel):
	# get the spatial dimensions of the image and the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	# allocate memory for the output image and 
	# pad the borders of the input image
	# to avoid spatial size reduction
	pad = int((kW -1) / 2)
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

	# loop over the input image, sliding the kernel across each coordinate
	# from left to right, top to bottom
	for y in np.arange(pad, iH+pad):
		for x  in np.arange(pad, iW+pad):
			# extract the region of interest of the image by extracting
			# the center region of the current x,y coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# perform the convolution by taking the 
			# element-wise multiplication between
			# the ROI and the kernel, then summing 
			# the matrix
			k = (roi * kernel).sum()


			# store the convolved value in the roi
			# coordinate of the output image 
			output[y - pad, x - pad] = k

	# return the output image
	output = rescale_intensity(output, in_range = (0, 255))
	output = (output * 255).astype("uint8")	
	return output

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype = "float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype = "float") * (1.0 / (21* 21))

# construct a sharpening filter
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype = "int")

# construct the Laplacian kernel for edge-detection
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype = "int")

# construct the Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")
 
# construct the Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")

# create a list of kernels to be applied to the image
# using the convolve function and openCV's filter2D
kernelBank = (
	("small_blur", smallBlur),
	("large_blur", largeBlur),
	("sharpen", sharpen),
	("laplacian", laplacian),
	("sobel_x", sobelX),
	("sobel_y", sobelY))

# load the input image and convert to grayscale 
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelName, kernel) in kernelBank:
	# apply the kernel to the grayscale image using 
	# the custom convolve function and OpenCV's 
	# filter2D
		# break
	# import code; code.interact(local=dict(globals(), **locals()))
	print("[INFO] applying {} kernel".format(kernelName))
	convolveOutput = convolve(gray, kernel)
	opencvOutput = cv2.filter2D(gray, -1, kernel)

	# show the output images
	print(convolveOutput)
	cv2.imshow("original", gray)
	cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
	cv2.imshow("{} - openCV".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
