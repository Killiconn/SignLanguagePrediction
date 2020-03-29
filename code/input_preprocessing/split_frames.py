# Program takes in a video and a number. The video is split into frames, the frames are resized and padded if need be.
import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_into_frames(filename, splitter):
	video = cv2.VideoCapture(filename)

	#Create empty numpy array to insert frames
	frameAmount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	array_width = 150
	array_height = 150

	img_array = np.empty((frameAmount, array_height, array_width), np.dtype('float64'))

	i = 0
	while i < frameAmount:
		cond, frame = video.read()
		
		#Once it hits the end break
		if cond == False: break

		#Convert to greyscale

		frame = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
		height, width = frame.shape
		#For Portrait videos
		if height > width:
			frame = frame[100:height-100,:]
		#For Landscape videos
		elif width > height:
			frame = frame[:,100:height-100]

		frame = add_padding(frame, array_height)  # Necessary input size for CNN
		
		img_array[i] = frame
	
		i += 1
	#the parameter, splitter, is to split the array. Since every frame cannot be taken,
	# the array is iterated over pending on this parameter.
	return img_array[::int(splitter)]

def add_padding(img, desired_size = 150):
	
	old_size = img.shape[:2] # old_size is in (height, width) format

	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])

	# new_size should be in (width, height) format

	im = cv2.resize(img, (new_size[1], new_size[0]))

	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)


	return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)