# Use this along with another program to introduce noise into the images in order to increase variance in data and increase the generalising ability of the models
# Different to other other serialisation data algorithms as it creates even ISL data splits data into training data and testing data with noise introduced
# This program also defines a class which can be used by other serialisation classes to create training and testing data

import numpy as np
import os
import cv2
from tqdm import tqdm # shows a progress bar for an iteration while it's executing
import random
from reduce_image_size import find_largest_dimension
from reduce_image_size import reduce_image_size
from tensorflow.keras.utils import to_categorical
import sys

# Class which encapsulates a data handler system to produce arrays of numpy matrices representing images and their labels
# Produces datasets which have uniform letter distribution 
class Data_Handler:
    def __init__(self, DATADIR):
        # Location of images passed to Data_Handler when initialised
        self.DATADIR = DATADIR

        # Relative locations of different image datasets
        self.DIRECTORIES = ["extracted_data_1", "extracted_data_2", "extracted_data_3", "extracted_data_4", "extracted_data_5", "extracted_data_6"] # Relative file locations of small ISL alphabet datasets

        # All labels to train the models to predict
        self.CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y", "Noise"]  # Removed J, X and Z as they involve motion

        # Interger version of CATEGORIES
        self.INT_LABELS = [i for i in range(24)] # Adding noise / no hands as a label --> label 23

        # Used in self.find_dim() to count number of for each letter in training set
        self.count_training_letters = [0] * 24

        # Used in self.find_dim() to count number of for each letter in testing set
        self.count_testing_letters = [0] * 24

        # Where training data will be saved to
        self.training_set = []

        # Where testing data will be saved to
        self.testing_set = []

        # Creates one-hot encoded version of INT_LABELS i.e. label for A, 0 --> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.categorical_labels = to_categorical(self.INT_LABELS, num_classes=None)

        # Used in conjunction with self.train_count below to restrict number of images per letter to this number
        # This ensures even data in training set
        self.restrict_training_letters = 0

        # Both are calculated in self.find_dim() 
        # Same as above just for testing data
        self.restrict_testing_letters = 0

        # Used by add_noise.py for cropping
        # Maximum length of a hand in all images, calculated in self.ind_dim()
        self.max_dim = 0

        # Used in create_data to ensure all letters have an even distribution of images in training set 
        self.train_count = [0] * 24

        # Used in create_data to ensure all letters have an even distribution of images in testing set
        self.test_count = [0] * 24

    # Extract information on the data
    # Sets up all the class variables needed for self.create_data()
    def find_dim(self):
        # Find largest dimenesion
        max_dim = 0
        for directory in self.DIRECTORIES:
            tmp_path = os.path.join(self.DATADIR, directory)  # Create path to directory
            direct_num = int(directory.split("_")[-1])        # Determine which directory it is to know where to store the data

            for category in self.CATEGORIES[:-1]:             # alphabet not including Noise label
                path = os.path.join(tmp_path, category)       # create path to images
                
                # extracted_data reserved for testing
                if direct_num < 6:
                    # count number of images in each letter in training set
                    self.count_training_letters[self.CATEGORIES.index(category)] += len(os.listdir(path))

                else:
                    # count number of images in each letter in testing set
                    self.count_testing_letters[self.CATEGORIES.index(category)] += len(os.listdir(path))

                # Go through every image and determine the largest hand dimension in the whole dataset
                for img in tqdm(os.listdir(path)):
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                    temp = find_largest_dimension(img_array)
                    if temp > max_dim:
                        max_dim = temp

        # Display results to the prompt
        print(self.count_training_letters)
        print(self.count_testing_letters)

        # Get minimum amount images so all will be even
        self.restrict_training_letters = min(self.count_training_letters[:-1]) # Don't include last label as that is noise and will be zero
        self.restrict_testing_letters = min(self.count_testing_letters[:-1])

        self.max_dim = max_dim

    # Create training and testing datasets
    def create_data(self, over_cond=False):
        # Can enable oversampling of some letters
        oversample = {} # default value, prevents oversampling unless changed
        if over_cond:
            # Which letters to oversample
            oversample_letters = [4, 5, 12, 18, 19] # ["E", "F", "N", "T", "U"] Letters the model is getting wrong, use oversampling to combat this
            self.restrict_training_letters -= 100   # force oversampling by reducing the number of images in other letters
            for letter in oversample_letters:
                oversample[letter] = True

        # Count of how many images are in each letter, used for troubleshooting and to implement even data
        self.train_count = [0] * 24
        self.test_count = [0] * 24

        # Go through every image in every letter category in every category
        for directory in self.DIRECTORIES:
            tmp_path = os.path.join(self.DATADIR, directory)  # Create path to directory
            direct_num = int(directory.split("_")[-1])        # Determine which directory it is to know where to store the data

            for category in self.CATEGORIES[:-1]:             # alphabet not including Noise label as noise is in a seperate location
                path = os.path.join(tmp_path, category)       # create path to images
                class_num = self.categorical_labels[self.CATEGORIES.index(category)] # One-hot encoded class label
                letter = self.CATEGORIES.index(category)                             # integer label
                
                # Oversample Case if oversample letter and not testing
                if (letter in oversample) and (direct_num < 6):
                    print("Oversampling: ", self.CATEGORIES[letter])
                    for img in os.listdir(path):
                        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # read image into array
                        new_array = reduce_image_size(img_array, self.max_dim)               # reduce the size of the image array from (480, 640) to (150, 150)
                        self.training_set.append([new_array, class_num])                     # add this to training_data
                        self.train_count[letter] += 1                                        # Count num letters for checks

                # Even Letter Distribution Case: if not extracted data 6 and not an oversampling letter
                elif direct_num < 6:
                    print("Adding: ", self.CATEGORIES[letter])
                    i = 0
                    # for every image in the directory as long as the max amount of images for each letter hasn't been reached for that letter
                    while i < len(os.listdir(path)) and (self.train_count[letter] < self.restrict_training_letters): # iterate over each image of hand signal
                        img = os.listdir(path)[i]
                        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                        new_array = reduce_image_size(img_array, self.max_dim)
                        self.training_set.append([new_array, class_num])                      # add this to training_data
                        self.train_count[letter] += 1                                         # remember amount of letters
                        i += 1

                # Testing Set Case
                else: 
                    print("Creating testing set")
                    for img in tqdm(os.listdir(path)):
                        if self.test_count[letter] < self.restrict_testing_letters:                    # Restrict letters some number of letters in testing data is homogenous
                            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)       # convert to array
                            new_array = reduce_image_size(img_array, self.max_dim)                     # resize image
                            self.testing_set.append([new_array, class_num])                            # add this to our testing_data
                            self.test_count[letter] += 1

            # print results to ensure even distribution of data
            print(self.train_count)
            print(self.test_count)

    def add_noise_images(self, parent_directory, file, letter=23):
        # Option to add noise data
        # This is data which has no hands in it
        # letter is integer value representing index of 'letter' in CATEGORIES, in this case NOISE

        print("Adding noise")

        path = os.path.join(self.DATADIR, parent_directory, file)        # Path to images            
        class_num = self.categorical_labels[letter]                      # Get one-hot encoded class number from categorical_labels using OWN_DATA dictionary values
        print(class_num)
        for img in tqdm(os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            if random.random() > 0.1:                             # Randomly take 90% images for training and 10% for testing 
                self.training_set.append([img_array, class_num])  # Add image to training data
                self.train_count[letter] += 1                     # Increment image counter
            else:
                self.testing_set.append([img_array, class_num])
                self.test_count[letter] += 1
       
        # print results to ensure even distribution of data
        print(self.train_count)
        print(self.test_count)

    def get_training(self):
        return self.training_set[:]  # return copy of training data

    def get_testing(self):
        return self.testing_set[:]  # return copy of testing data


def main():
    # Taking diectory off the commandline
    # Killian --> "/home/killian/Documents/3rdYearPro/Person1"
    # Bill    --> "C:\\Users\\Bill\\3D Objects\\cnn_datasets\\ISL"

    DATADIR = str(sys.argv[1])

    data_handler = Data_Handler(DATADIR)                          # Create data handler
    data_handler.find_dim()                                       # Find the largest hand. This is used by add_noise.py to crop the images appropriately ( from (480,640) into (150, 150))
    data_handler.create_data(over_cond=False)                     # Create training and testing data
    data_handler.add_noise_images("own_extracted_data", "Noise")  # Add noise images to datasets. This is a label which represents images with no hands in them

    training_set = data_handler.get_training() # get data from data handler
    testing_set = data_handler.get_testing()

    # Shuffle training data
    random.shuffle(training_set)

    X = []
    y = []

    # Extract labels and features from training data set
    for features, label in training_set:
        X.append(features)
        y.append(label)

    # Changes X to a numpy array
    # X will have the shape (?, 150, 150, 1), where ? is the number of images in the np array, (150, 150) is the image dimensions and 1 shows that these are greyscale images ( not rgb )
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

    os.chdir(DATADIR) # Change current working directory

    # Create serialised training datasets
    pickle_out = open("X_even.pickle","wb")
    joblib.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y_even.pickle","wb")
    joblib.dump(y, pickle_out)
    pickle_out.close()

    # Create serialised testing datasets
    pickle_out = open("X_test_even.pickle","wb")
    joblib.dump(X_test, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test_even.pickle","wb")
    joblib.dump(y_test, pickle_out)
    pickle_out.close()

if __name__ == '__main__':
    main()
