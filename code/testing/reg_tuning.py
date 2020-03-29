# This program tests multiple different l2 regulisation values and outputs the effect of these different values on the model

import os
import sys
import joblib
import os
from convolutedNN import create_model

# Changing working directory to find pickle files
# Taken off the commandline to reduce merge conflicts
# Taking directory off the commandline
# Killian --> "/home/killian/Documents/3rdYearPro/Person1"
# Bill    --> "C:\Users\Bill\3D Objects\cnn_datasets\ISL"

DATADIR = sys.argv[1]
os.chdir(DATADIR)

#Load in training files
pickle_in = open("X_train.pickle","rb")
X_train = joblib.load(pickle_in)
print("JOBLIB X done")


pickle_in = open("y_train.pickle","rb")
y_train = joblib.load(pickle_in)
print("JOBLIB y done")

#Load in testing files
pickle_in = open("X_test.pickle","rb")
X_test = joblib.load(pickle_in)
print("JOBLIB X done")


pickle_in = open("y_test.pickle","rb")
y_test = joblib.load(pickle_in)
print("JOBLIB y done")

#Normalise the data by scaling(diving by pixel size)
X_train = X_train/255.0
X_test = X_test/255.0

# Testing different regulisation sizes to find the best [0.0001, 0.001]
reg_size = [0.0002, 0.0004, 0.0006, 0.0008, 0.001]

for reg in reg_size:
	model = create_model(X_train.shape[1:], 0.6, reg) # craete model
	model.fit(X_train, y_train, batch_size=16, epochs=2, validation_split=(X_test, y_test)) # train
	print("This model had a reg value of: {:}".format(reg)) # output results
