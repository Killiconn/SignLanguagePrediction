# This program tests multiple different dropout rates and outputs the effect of these different rates on the model

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

# Testing different dropout values to find the best [0.0, 1)
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for rate in dropout_rate:
    model = create_model(X_train.shape[1:], rate) # create the model
    model.fit(X_train, y_train, batch_size=16, epochs=2, validation_data=(X_test, y_test)) # train 
    print("This model had a dropout rate of: {:}".format(rate)) # output the result
