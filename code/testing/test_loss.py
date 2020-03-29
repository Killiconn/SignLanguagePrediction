'''
This program defines a custom_loss function which was used in an attempt to improve the accuracy of the models
This program only exists to test the loss function on sample hard coded values which may appear during training
Running the program with diffrent pararmeters shows how the loss function behaves on different letters

This loss function adds a penalty to the already existing loss function categorical_crossentrophy
Take from the technical manual:
"The added penalty was calculated based on a predicted labels perceived distance from the true label.
For instance, if the true value was an "A", but the predicted value was a "R", then the loss function would add a larger penalty than if the predicted value was a "C".
The thinking behind this is that the penalty will push the predictions closer to the true labels. However, this loss function did not succeed in increasing the accuracy of the model."
'''

import tensorflow as tf
from tensorflow.keras import backend as K
import os
import sys
import joblib
import numpy as np

# Writing a custom loss function
def custom_loss(y_true, y_predict):
    # y_true is the shape of the last leyer of the model i.e. (?, 23), where ? is the batch size
    # y_pred will have shape (?, 23), same as y_true
    # Both y_pred and y_true are tensors
    # Want to use categorical crossentrophy but with a penalty for being far from actual answer
    loss_variable = calculate_new_loss_variable(y_true, y_predict)
    return K.categorical_crossentropy(y_true, y_predict)  + loss_variable

def calculate_new_loss_variable(y_true, y_pred):
    y = K.argmax(y_true, axis=1)     # Array of all the indexes of max values for true values
    y_hat = K.argmax(y_pred, axis=1) # Array of all the indexes of max values for predictions
    
    tmp = K.cast(K.sum(K.abs(y - y_hat)), dtype='float32')
    loss =  tmp * 0.001 # [0, 0.506]
    return loss

def main():
    # This is only called when testing the loss functions capabilities

    # A the max value --> guessed A
    y_hat = [[0.30674765, 0.01799027, 0.07322054, 0.04238936, 0.00807157, 0.03875672, 0.02900247, 0.0618128,  0.0276928,  0.01263931, 0.03060391, 0.02297725, 0.09377529, 0.12260796, 0.00835913, 0.00694755, 0.01799027, 0.00486686, 0.03337229, 0.00969546, 0.01395214, 0.00624659, 0.0170279]]

    y_label = [([0]*23)]

    # Assume A is the right answer
    y_label[0][1] = 1

    y_hat = tf.convert_to_tensor(y_hat, dtype=tf.float32)
    y_label = tf.convert_to_tensor(y_label, dtype=tf.float32)

    sess = tf.Session()
    result_tensor = custom_loss(y_label, y_hat)
    with sess.as_default():
        print(result_tensor.eval())

if __name__ == '__main__':
    main()
