'''
This program defines a model class which defines a CNN from scratch with variable hyperparameters and layers
The amount of regularisation (dropout values and l2 values) is variable,
along with the number of dense and convolutional layers as well as the size of the inner dense layers.
There is a basic minimum skeleton of a CNN if all values are set it 0
Similiar to other programs just way more variable
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
import os
import sys
import joblib
import os

# Variable CNN model from scratch defined using the Keras API with tensorflow backend
class model:
    def create_model(shape, layer_size=16, num_conv_layers=0, num_dense_layers=0, dropout_rate=0.6, reg=0.0006): # Ran evaluation test and results showed that 0.6 and 0.0006 are optimum values
        model = Sequential()
        print("Created model")

        # First Layer - Don't touch
        model.add(Conv2D(16, (3, 3), input_shape=shape)) # Convolution Layer (Units/filters, Window/Kernal Size, Input Shape)
        model.add(Activation('relu'))                    # right Linear activation function
        model.add(MaxPooling2D(pool_size=(2, 2)))        # max pool
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

        # Output Layer - outputs from the previous layers fed to this layer
        model.add(Dense(26))
        model.add(Activation('softmax')) # Convert the values in the  output layer to follow a probability distribution
        print("Dense Layer Done")

        # compile the model with loss function optimiser and metric
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # prints a summary of the model on the commandline
        model.summary()

        return model

def main():
    # Changing working directory to find pickle files
    # Taking directory of serialised datasets from commandline

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

    

    model = create_model(X_train.shape[1:], )

    model.summary()

    model.fit(X_train, y_train, batch_size=16, epochs=1, validation_data=(X_test, y_test))

    # epoch is one complete presentation of the data set to be learned

    model.save("model.h5")
    del model

if __name__ == '__main__':
    main()