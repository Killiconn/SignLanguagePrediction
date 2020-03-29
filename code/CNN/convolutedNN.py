import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
import os
import sys
import joblib
import os

def create_model(shape, dropout_rate=0.6, reg=0.0006): # Ran evaluation test and results showed that 0.6 and 0.0006 are optimum values
    model = Sequential()
    print("Created model")

    # First Layer
    model.add(Conv2D(16, (7, 7), input_shape=shape, kernel_regularizer=l2(reg))) 
    model.add(Activation('relu')) # right Linear
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("First Layer Done")

    # Second Layer
    model.add(Conv2D(32, (5,5), strides=(2,2), kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    # model.add(BatchNormalization())

    # Third Layer
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    # this converts our 2D feature maps to 1D feature
    model.add(Flatten()) 
    model.add(Dense(32, kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))

    print("Flatten Layer Done")

    # Use Dropout to prevent overfitting
    model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(26))
    # feeds all outputs from the previous layer to all its neurons
    model.add(Activation('softmax'))
    print("Dense Layer Done")


    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model

def main():
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

    # Convolution Layer (Units/filters, Window/Kernal Size, Input Shape)
    # filters determines the number of kernels to convolve with the input volume.

    model = create_model(X_train.shape[1:], )

    model.summary()
    
    i = 0
    while i < 10: # Number of epochs to run the model for at every iteration
        # Train and evaluate model
        print("continue? [y/n]") # check if user still wants to tarin model
        answer = input()
        # if so keep training the model
        if answer.strip() == "y":
            model.fit(X_train, y_train, batch_size=16, epochs=1, validation_data=(X_test, y_test))
        # otherwise stop training
        else:
            break
        i += 1
        print("Number of epochs passed: {:}".format(i))

    # epoch is one complete presentation of the data set to be learned

    # save the model to current working directory
    model.save("model.h5")
    del model

if __name__ == '__main__':
    main()