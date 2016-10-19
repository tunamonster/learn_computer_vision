# tutorial here https://realpython.com/blog/python/face-recognition-with-python/
import argparse
import cv2

# construct arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to image")
ap.add_argument("-c", "--cascade", help = "path to the cascade model data")
args = vars(ap.parse_args())

# create the haar cascade
face_cascade = cv2.CascadeClassifier(args["cascade"])

# read the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the image
faces = face_cascade.detectMultiScale(
	# pass the image to run detection on
	gray,
	# account for difference in face size 
	# for example due to camera distance
	scaleFactor = 1.1,
	# number of objects to be detected next to each other
	# before a face is detected 
	minNeighbors = 5,
	# minimimum window size
	minSize = (30,30)
	# flags = cv2.CV_HAAR_SCALE_IMAGE
	)

print("found {0} faces!".format(len(faces)))

# draw a rectangle around each face 
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)