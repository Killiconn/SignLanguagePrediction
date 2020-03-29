# A larger variable CNN which allows for kernel initialiser hyperparameter tuning
# Used by kernel_tuning.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
import os
import sys
import joblib
import os
import random

# Function which creates and returns a compiled Keras model
def create_model(shape, dropout_rate=0.5, reg=0.0005, ker_init="he_uniform"): # Ran evaluation test and results showed that 0.6 and 0.0006 are optimum values
    model = Sequential()
    print("Created model")

    # First Layer
    model.add(Conv2D(32, (7, 7), input_shape=shape, kernel_regularizer=l2(reg), kernel_initilizer=ker_init)) 
    model.add(Activation('relu')) # right Linear
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Layer
    model.add(Conv2D(64, (5, 5), kernel_regularizer=l2(reg), kernel_initilizer=ker_init))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    # model.add(BatchNormalization())

    # Third Layer
    model.add(Conv2D(128, (3, 3), kernel_regularizer=l2(reg), kernel_initilizer=ker_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Fourth Layer
    model.add(Conv2D(256, (3, 3), kernel_regularizer=l2(reg), kernel_initilizer=ker_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    # this converts our 2D feature maps to 1D feature
    model.add(Flatten()) 
    model.add(Dense(128, kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))

    # Use Dropout to prevent overfitting
    model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(26))
    # feeds all outputs from the previous layer to all its neurons
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Model compiled succesfully')

    model.summary()

    return model

def main():
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
    X_test_1 = joblib.load(pickle_in)
    print("JOBLIB X done")


    pickle_in = open("y_test_1.pickle","rb")
    y_test_1 = joblib.load(pickle_in)
    print("JOBLIB y done")

    #Normalise the data by scaling(diving by pixel size)
    X_train = X_train/255.0
    X_test_1 = X_test_1/255.0

    model = create_model(X_train.shape[1:]) # create the model

    model.summary() # print summary of the model

    model.fit(X_train, y_train, batch_size=26, epochs=10, validation_data=(X_test_1, y_test_1)) # train the model

    # save the model to current working directory
    model.save("model.h5")
    del model

if __name__ == '__main__':
    main()