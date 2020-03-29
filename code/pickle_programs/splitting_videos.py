#This program creates a dataset of images using the frames of a video.
import cv2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#The over sampled letters that required a bigger dataset was N, A, T, U, F and Noise.
def split_vid_into_frames():
	oversampledLetters_Vids = ["VideoOfN.mp4", "VideoOfA.mp4", "VideoOfT.mp4", "VideoOfU.mp4", "VideoOF_F.mp4", "Noise.mp4"]
	oversampledLetters_Folders = ["N_OS", "A_OS", "T_OS", "U_OS", "F_OS", "Noise"]

	for vid in oversampledLetters_Vids:
		vidcap = cv2.VideoCapture('/home/killian/Documents/3rdYearPro/oversampledHomemadeData/' + vid)
		#read in each frame for the video
		success,image = vidcap.read()
		count = 0

		#Convert to grayscale and resize the image.
		while success:
		  image = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
		  height, width = image.shape
		  image = image[:,20:]
		  image = cv2.resize(image, (150, 150))

		  #Write the modified images to a directory
		  cv2.imwrite("/home/killian/Documents/3rdYearPro/oversampledHomemadeData/" + oversampledLetters_Folders[oversampledLetters_Vids.index(vid)] + "/frame%d.jpg" % count, image)     # save frame as JPEG file      
		  success,image = vidcap.read()

		  print('Read a new frame: ', success)
		print("Done" + oversampledLetters_Folders[oversampledLetters_Vids.index(vid)])

