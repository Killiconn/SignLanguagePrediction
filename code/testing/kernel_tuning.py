# Program to test different kernel initialisers on the model
# the results will be outputted n the commandline

import os
import sys
import joblib
import os
from larger_variable_cnn import create_model

# Changing working directory to find pickle files
# Taken off the commandline to reduce merge conflicts
# Taking diectory off the commandline
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
pickle_in = open("X_test_1.pickle","rb")
X_test = joblib.load(pickle_in)
print("JOBLIB X done")


pickle_in = open("y_test_1.pickle","rb")
y_test = joblib.load(pickle_in)
print("JOBLIB y done")

#Normalise the data by scaling(diving by pixel size)
X_train = X_train/255.0
X_test = X_test/255.0

# model = create_model(X_train.shape[1:])

# Testing different kernel initialisers to find the best
# These values are used to instanciate the weights of the model
# These are differnt number distributions, and the weights will be any vaues in these distributions 
kernels = ["he_uniform", "truncated_normal", "random_uniform", "glorot_uniform"] # Default is glorot_uniform and also the best

for kernel in kernels:
	model = create_model(X_train.shape[1:], ker_init=kernel) # create model
	model.fit(X_train, y_train, batch_size=16, epochs=2, validation_data=(X_test, y_test)) # train
	print("This model had a kernel value of: {:}".format(kernel)) # output test results

