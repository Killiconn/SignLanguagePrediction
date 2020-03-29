# Program similiar to program which creates even datasets except this program introduces more oversampling for specific letters
# Also includes data which has been create by ourselves to hopefully improve accuracy of certaiin letters
# Namely A, F, N, T, U

import numpy as np
import os
import cv2
from tqdm import tqdm # shows a progress bar for an iteration while it's executing
import random
from reduce_image_size import find_largest_dimension
from reduce_image_size import reduce_image_size
from tensorflow.keras.utils import to_categorical
import sys
from create_even_data import Data_Handler
from tensorflow.keras.utils import to_categorical

class Oversample_Data_Handler(Data_Handler):
    def __init__(self, DATADIR):
        # Look at create_even_data.py for all class variables

        super().__init__(DATADIR)

        self.OWN_DATA = "own_extracted_data" # This data is already changed to be (150, 150)
        
        # Categories of letters to be sampled
        self.OVER_CAT = {
            "F":5,
            "N":12,
            "T":18,
            "U":19,
            "Noise":23}

    def add_own_data(self):
        '''
        Add created data to training set
        These are images which have been created by ourselves
        Therefore can't use create_data as this has been tailored for the dataset provided by Alstair Sutherland
        Dataset --> (https://github.com/marlondcu/ISL/tree/master/Frames?fbclid=IwAR3921N7ApZR-X4eFDvYfylYFLjOYo5aCJvjnnVFHY5_92WMn_NwP-qx9SY)
        '''
        path = os.path.join(self.DATADIR, self.OWN_DATA)
        for file in os.listdir(path):
            if file in self.OVER_CAT:                       # Don't want to always oversample these letters, determined in self.OVER_CAT if it is to be oversampled
                new_path = os.path.join(path, file)         # full path to image
                letter = self.OVER_CAT[file]                # index of label in self.CATEGORIES; defined in parent class
                class_num = self.categorical_labels[letter] # Get one-hot encoded class number from categorical_labels using OWN_DATA dictionary values / letter
                
                for img in tqdm(os.listdir(new_path)):
                    img_array = cv2.imread(os.path.join(new_path,img), cv2.IMREAD_GRAYSCALE) # convert image to an array
                    self.training_set.append([img_array, class_num])                         # Add image to training data
                    self.train_count[letter] += 1                                            # Increment image counter
        
        # print results to ensure oversampling worked
        print(self.train_count)

    def add_noise(self):
        self.add_noise_images(self.OWN_DATA, "Noise")  # Call method in parent calss to add noise in both training and testing data

def main():
    # Taking directory off the commandline
    # Killian --> "/home/killian/Documents/3rdYearPro/Person1"
    # Bill    --> "C:\\Users\\Bill\\3D Objects\\cnn_datasets\\ISL"

    DATADIR = str(sys.argv[1]) # directory location of parent directory for all images

    data_handler = Oversample_Data_Handler(DATADIR)
    data_handler.find_dim()
    data_handler.create_data(over_cond=True) # enable oversampled
    data_handler.add_own_data()
    data_handler.add_noise()    # add noise 

    # get data
    training_set = data_handler.get_training()
    testing_set = data_handler.get_testing()

    # Shuffle training data
    random.shuffle(training_set)

    X = []
    y = []

    for features, label in training_set:
        X.append(features)
        y.append(label)

    # Changes X to a numpy array and palces pixel values in individual lists of one value contained within their own image lists
    X = np.array(X).reshape(-1, 150, 150, 1)
    y = np.array(y)

    # Repeat for testing data
    X_test = []
    y_test = []

    for features, label in testing_set:
        X_test.append(features)
        y_test.append(label)

    X_test = np.array(X_test).reshape(-1, 150, 150, 1)
    y_test = np.array(y_test)


    import joblib

    os.chdir(DATADIR) # Change current working directory to parent directory of images

    # Serialise data
    pickle_out = open("X_oversample.pickle","wb")
    joblib.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y_oversample.pickle","wb")
    joblib.dump(y, pickle_out)
    pickle_out.close()

    pickle_out = open("X_test_oversample.pickle","wb")
    joblib.dump(X_test, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test_oversample.pickle","wb")
    joblib.dump(y_test, pickle_out)
    pickle_out.close()

if __name__ == '__main__':
    main()

