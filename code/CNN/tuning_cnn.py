'''
This program is very similiar to other variable CNN programs we've written
However this program was used to multiple dense layer, convolution layer, and layer size combinations to the best model after ten epochs
The results from this test was stored using tensorboard callbacks
The results can be seen in the logs file within testing repo within the code repo on gitlab
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import os
import sys
import joblib
import os

# Program to test and tune multiple variables of cnn

# Changing working directory to find pickle files

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

# Different variables to be tested
dense_layers = [0,1,2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

class model:
    def create_model(shape, layer_size=16, num_conv_layers=0, num_dense_layesr=0, dropout_rate=0.6, reg=0.0006): # Ran evaluation test and results showed that 0.6 and 0.0006 are optimum values
        model = Sequential()
        print("Created model")

        # First Layer - Don't touch
        model.add(Conv2D(16, (3, 3), input_shape=shape)) 
        model.add(Activation('relu')) # right Linear
        model.add(MaxPooling2D(pool_size=(2, 2)))
        print("First Layer Done")

        # Other CNN Layers
        for l in range(num_conv_layers-1):
            model.add(Conv2D(layer_size, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        # this converts our 2D feature maps to 1D feature - Don't touch
        model.add(Flatten())

        for _ in range(num_dense_layers):
            model.add(Dense(layer_size))
            model.add(Activation('relu'))

        # Use Dropout to prevent overfitting
        model.add(Dropout(dropout_rate))

        # Output Layer
        model.add(Dense(26))
        # feeds all outputs from the previous layer to all its neurons
        model.add(Activation('softmax'))
        print("Dense Layer Done")


        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        return model

# test every combination of number of dense layers with every layer size and every number of covolutional layers as described in the lists above
for dense_layer in dense_layers:
    # for every number of dense layers in list dense_layers
    for layer_size in layer_sizes:
        # for every layer size in layer_sizes
        for conv_layer in conv_layers:
            # for every number of convolution layers in conv_layers
            NAME = "{}-conv-{}-nodes-{}-dense".format(conv_layer, layer_size, dense_layer) # custom name of model
            print(NAME)

            model = model() # instansiate model

            model = model.create_model(layer_size=layer_size, num_conv_layers=conv_layer, num_dense_layers=dense_layer)  # Create model

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME)) # use tensorboard to log models

            model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard]) # train model

# access logs with following line
# tensorboard --logdir='./logs' --host=127.0.0.1
