import numpy as np 
import cv2

def order_points(pts):
	# initialize the x,y of 4 corners, going clockwise from top left
	rect = np.zeros((4,2), dtype = "float32")

	# top left point has the smallst sum, bottom right has largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# top right has the smallest difference,
	# bottom left has the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	rect = order_points(pts)
	(top_left, top_right, bottom_right, bottom_left) = rect

	# compute the width of the new image, which is the
	# maximum geometric distance between (bottom_right and bottom_left)
	# and (top_right and top_left)
	width_bottom = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + (bottom_right[1] - bottom_left[1]) ** 2 )
	width_top = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2 ))
	# break
	import code; code.interact(local=dict(globals(), **locals()))
	width_image = max(int(width_bottom), int(width_top))

	# compute the height of the new image, which is the maximum
	# geometric distance between (top_left and bottom_left) and 
	# (top_right and bottom_right)
	height_left = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
	height_right = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
	height_image = max(int(height_left), int(height_right))

	# construct the destination points to obtain a top-down view 
	# of the image, going clockwise from top-left
	dst = np.array([
		[0, 0],
		[width_image - 1, 0],
		[width_image - 1, height_image - 1],
		[0, height_image - 1]], dtype = "float32")

	# compute and apply the perspective transform matrix
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (width_image, height_image))	

	# return the warped image
	return warped	