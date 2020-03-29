'''
This is a program to test all the different pretrained models which come with the Keras API
The models are tested on a basic model and the results recorded by way of callbacks and displayed with Tensorboard
'''

# Load in pretrained models
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.nasnet import NASNetMobile

# Load in necessary libraries
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import os
import sys
import joblib
import numpy as np

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
pickle_in = open("X_test.pickle","rb")
X_test = joblib.load(pickle_in)
print("JOBLIB X done")


pickle_in = open("y_test.pickle","rb")
y_test = joblib.load(pickle_in)
print("JOBLIB y done")

#Normalise the data by scaling(diving by pixel size)
X_train = X_train/255.0
X_test = X_test/255.0

# height and width of inputed data
height, width = X_test.shape[1:3]

# Pretrained model names
pretrained_models = ["MobileNetV2", "Xception", "ResNet50", "InceptionV3", "InceptionResNetV2", "DenseNet121", "NASNetMobile", "VGG19"]

# Evalualte effectiveness of each model on predicting data
for pre_model in pretrained_models:
    try:
        model = Sequential() 
        model.add(Conv2D(3, (3,3), padding='same', input_shape=X_test.shape[1:]))                             # Layer to convert data from greyscale to RGB as pretrained models are trained on coloured data
        base_model = eval(pre_model)(weights='imagenet', include_top=False, input_shape=(height, width, 3))   # Defining pretrained model
        model.add(base_model)                                                                                 # adding model to simple architecture
        model.add(MaxPooling2D(pool_size=(2, 2)))                                                             # max pool output of pretrained model
        model.add(Flatten())                                                                                  # flatten into one dimensional array
        model.add(Dense(24, activation='softmax'))                                                            # fully connected output layer

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])                # compile model

        tensorboard = TensorBoard(log_dir="logs/pretrained/{}".format(NAME))                                  # define folder to save tensorboard logs

        print("Testing ", pre_model)
        model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=1, validation_data=(X_test, y_test), callbacks=[tensorboard]) # train the model

    except:
        print(pre_model, "caused an error")

