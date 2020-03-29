# This program tests a data set against one particluar model
from tensorflow.keras.models import load_model
import cv2
import joblib
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

# model_location = "C:\\Users\\Bill\\3D Objects\\2020-ca326-kconnolly-signlanguagetranslator\\code\\CNN"
# model_location = '/home/killian/Documents/3rdYearPro/2020-ca326-kconnolly-signlanguagetranslator/code/CNN/model.h5'

# Loading model / CNN
model = load_model('/home/killian/Documents/3rdYearPro/model_pre_25_02_2020.h5')

# take images location as input
pickle_location = input()
os.chdir(pickle_location)

#Load in pickle files
pickle_in = open("y_test_no_motion.pickle","rb")
X = joblib.load(pickle_in)
print("JOBLIB X done")
# X.shape should be (8864, 154, 154, 1)


pickle_in = open("y_test_no_motion.pickle","rb")
y = joblib.load(pickle_in)
print("JOBLIB y done")

test_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

X = X/255.0

model.evaluate(X, y, verbose=1)
