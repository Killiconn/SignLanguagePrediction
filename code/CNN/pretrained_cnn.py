import tensorflow as tf
from tensorflow.keras.applications.xception import Xception

# Load in necessary libraries
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GaussianNoise
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os
import sys
import joblib
import numpy as np


# Changing working directory to find pickle files
# Taking directory off the commandline
DATADIR = sys.argv[1]
os.chdir(DATADIR)

#Load in training files
pickle_in = open("X.pickle","rb")
X_train = joblib.load(pickle_in)
print("JOBLIB X done")


pickle_in = open("y.pickle","rb")
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

height, width = X_test.shape[1:3]

# Building the model
model = Sequential()
input_layer = Conv2D(3, (1,1), input_shape=X_test.shape[1:], kernel_initializer='ones')
input_layer.trainable = False
model.add(input_layer)
base_model = Xception(weights='imagenet', include_top=False, input_shape=(height, width, 3))
model.add(base_model)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.45)) # was 0.4
model.add(Dense(24, activation='softmax'))

print("Compiling...")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the summary of the model
model.summary()

print("Testing...")
model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1, validation_data=(X_test, y_test))#, callbacks=[tensorboard])

model.save("final_models/model_1.h5")
del model
