'''
Program to create radically smaller pickles
In this program the Data_Handler class is inherited and modified
This program should produce four training pickle files and four testing pickle files
This is in an effort to train smaller CNN models on less labels to hopefully improve accuracy
The four different classes of labels as follows:
["A", "B", "C", "D", "F"]
["E", "G", "H", "R", "K", "L"]
["M", "O", "P", "Q", "R", "S"]
["N", "T", "U", "V", "W", "Y"]
These were split apart based on their similiarities i.e. if two signs were similiar they were seperated.
These datasets will hopefully produce four models which will be combined to form a larger more accurate model
'''

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

class Smaller_Data_Handler(Data_Handler):
    def __init__(self, DATADIR, CAT):
        # Look at create_even_data.py for all class variables

        super().__init__(DATADIR)

        self.INT_LABELS = [i for i in range(len(CAT))] # now 6 labels
        
        self.categorical_labels = to_categorical(self.INT_LABELS, num_classes=None)

        print(self.categorical_labels)
        print(self.categorical_labels.shape)

        # only letters to be sampled in this program
        # created from inputted list
        self.cat_dict = {}
        for i in range(len(CAT)):
            key = CAT[i]
            value = self.INT_LABELS[i]
            self.cat_dict[key] = value
        
        # Need to define OTHER letters
        self.others = {}
        for i in range(len(self.CATEGORIES)): # from parent class
            key = self.CATEGORIES[i]
            if key not in self.cat_dict: # then add to others
                self.others[key] = i

        self.other_max_train = 0
        self.other_max_test = 0

    def create_small_data(self):
        # Same code as even just modified
        self.other_max_train = (self.restrict_training_letters) / (self.restrict_training_letters * 18)     # 18 other letters
        self.other_max_test = (self.restrict_testing_letters) / (self.restrict_testing_letters * 18)        # 18 other letters, percentage of images to take at random from massive datasets
        print(self.other_max_train)
        print(self.other_max_test) 
        for directory in self.DIRECTORIES:
            tmp_path = os.path.join(self.DATADIR, directory)  # Create path to directory
            direct_num = int(directory.split("_")[-1])        # Determine which directory it is to know where to store the data

            for category in self.CATEGORIES[:-1]:             # alphabet not including Noise label
                path = os.path.join(tmp_path, category)       # create path to images
                if category in self.cat_dict:                  # if one of labels
                    class_num = self.categorical_labels[self.cat_dict[category]] # One-hot encoded class label
                    letter = self.CATEGORIES.index(category)
                              # integer label
                else:
                    class_num = self.categorical_labels[6]  # other label
                    letter = self.others[category]
                print(class_num)
                if direct_num < 6:
                    # Even Letter Distribution Case
                    # Just add the letters same as always
                    # not other case
                    if category in self.cat_dict:
                        print("Adding: ", category)
                        i = 0
                        while i < len(os.listdir(path)) and (self.train_count[letter] < self.restrict_training_letters): # iterate over each image of hand signal
                            img = os.listdir(path)[i]
                            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                            new_array = reduce_image_size(img_array, self.max_dim)
                            self.training_set.append([new_array, class_num])  # add this to our training_data
                            self.train_count[letter] += 1
                            i += 1

                    else:
                        # Add all other letters to others dataset, and later on randomly select a certain amount of them
                        for img in tqdm(os.listdir(path)):
                            if random.random() > (1 - self.other_max_train):
                                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                                new_array = reduce_image_size(img_array, self.max_dim)
                                self.training_set.append([new_array, class_num])
                                self.train_count[letter] += 1

                # Testing Set Case
                else:
                    if category in self.cat_dict: # not other
                        print("Creating testing set")
                        for img in tqdm(os.listdir(path)):
                            if self.test_count[letter] < self.restrict_testing_letters:
                                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                                new_array = reduce_image_size(img_array, self.max_dim)
                                self.testing_set.append([new_array, class_num])                            # add this to our testing_data
                                self.test_count[letter] += 1

                    else:
                        for img in tqdm(os.listdir(path)):
                            if random.random() > (1 - self.other_max_test):
                                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                                new_array = reduce_image_size(img_array, self.max_dim)
                                self.testing_set.append([new_array, class_num])
                                self.test_count[letter] += 1
                

        print(self.train_count)
        print(self.test_count)
    

def main():
    # Taking diectory off the commandline
    # Killian --> "/home/killian/Documents/3rdYearPro/Person1"
    # Bill    --> "C:\\Users\\Bill\\3D Objects\\cnn_datasets\\ISL"

    DATADIR = str(sys.argv[1])

    ID = int(sys.argv[2])  # which pickle is it [0,3]

    '''
    ["A", "B", "C", "D", "F", "Noise"]
    ["E", "G", "H", "R", "K", "L"]
    ["M", "O", "P", "Q", "R", "S"]
    ["N", "T", "U", "V", "W", "Y"]
    '''

    CATEGORIES_lst = [["A", "B", "C", "D", "F", "Noise", "Other"],
    ["E", "G", "H", "I", "K", "L", "Other"],
    ["M", "O", "P", "Q", "R", "S", "Other"],
    ["N", "T", "U", "V", "W", "Y", "Other"],]

    CATEGORIES = CATEGORIES_lst[ID]  # which data to use

    data_handler = Smaller_Data_Handler(DATADIR, CATEGORIES)
    data_handler.find_dim()
    data_handler.create_small_data()
    
    if ID == 0:
        data_handler.add_noise_images("own_extracted_data", "Noise", letter=5)  # Add noise images to datasets

    training_set = data_handler.get_training()
    testing_set = data_handler.get_testing()

    # Shuffle training data
    random.shuffle(training_set)
    random.shuffle(testing_set)

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

    print(X.shape)
    print(y.shape)
    print(X_test.shape)
    print(y_test.shape)

    import joblib

    DATADIR = os.path.join(DATADIR, "smaller_pickles")
    os.chdir(DATADIR) # Change current working directory

    pickle_out = open("X_{:}.pickle".format(ID),"wb")
    joblib.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y_{:}.pickle".format(ID),"wb")
    joblib.dump(y, pickle_out)
    pickle_out.close()

    pickle_out = open("X_test_{:}.pickle".format(ID),"wb")
    joblib.dump(X_test, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test_{:}.pickle".format(ID),"wb")
    joblib.dump(y_test, pickle_out)
    pickle_out.close()

if __name__ == '__main__':
    main()

