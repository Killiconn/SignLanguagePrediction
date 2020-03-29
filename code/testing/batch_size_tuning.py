# This program tests multiple different batch sizes to output the effect of different batch sizes on the model

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

# model = create_model(X_train.shape[1:])

# Testing different batch sizes to find the best [16, 32]
batch_size = [16, 20, 24, 28, 32]

for size in batch_size:
    model = create_model(X_train.shape[1:], 0.3) # create the model
    model.fit(X_train, y_train, batch_size=size, epochs=2, validation_data=(X_test, y_test)) # train the model
    print("This model had a batch size of: {:}".format(size)) # output the rseult of the model

