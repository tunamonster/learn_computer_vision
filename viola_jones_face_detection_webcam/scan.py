# tutorial here: https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/
# modifications
# used half sized frames to reduce dropped camera frames
# added frame coordinates in view rectangle

import cv2
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", help="path to the cascade path")
args = vars(ap.parse_args())

video_stream = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(args["cascade"])

i = 0
while True:
	# capture frame by frame
	(ret, frame) = video_stream.read()
	# resize frame
	height, width = frame.shape[:2]
	frame = cv2.resize(frame, (int(width/2), int(height/2) ))
	# halve frame size to reduce processing lag


	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# pass face detection params
	faces = face_cascade.detectMultiScale(
		gray, scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (20,20)
		)

	# draw a rectangl around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
		# print text on video stream
		font = cv2.FONT_HERSHEY_SIMPLEX

		# declare corners and put them in frame
		top_left = (x, y)
		top_right = (x+w, y)
		bottom_left = (x, y+h) 
		bottom_right = (x+w, y+h)
		corners = [top_left, top_right, bottom_left, bottom_right]
		for corner in corners:
			cv2.putText(frame, str(corner), corner, font, 1, (0, 255, 0), 2)


	cv2.imshow('Video', frame)

	# break the loop on pressing 'q'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	i += 1
	print(i)
# close the video stream
video_stream.release()
cv2.destroyAllWindows()

