# Not changing the images at all
# Taking images from four people and using two for validation
# This was the original data serialising program
# Images don't get cropped or the ratio changed

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm # shows a progress bar for an iteration while it's executing
import random
from reduce_image_size import find_largest_dimension
from reduce_image_size import reduce_image_size
from tensorflow.keras.utils import to_categorical
import sys

# Taking directory off the commandline
# Killian --> "/home/killian/Documents/3rdYearPro/Person1"
# Bill    --> "C:\Users\Bill\3D Objects\cnn_datasets\ISL" 
DATADIR = str(sys.argv[1])

CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Will use extracted_data_5 and extracted_data_6 for validation
DIRECTORIES = ["extracted_data_1", "extracted_data_2", "extracted_data_3", "extracted_data_4", "extracted_data_5", "extracted_data_6"] # Relative file locations of small ISL alphabet datasets

INT_LABELS = [i for i in range(26)]

training_set = []
testing_set_1 = []
testing_set_2 = []

def create_training_data():
	# change labels to one-hot encoded
    categorical_labels = to_categorical(INT_LABELS, num_classes=None)

    # Want to take data from all of the ISL dataset
    for directory in DIRECTORIES:
        tmp_path = os.path.join(DATADIR, directory)  # Create path to directory
        direct_num = int(directory.split("_")[-1])   # Determine which directory it is to know where to store the data
        for category in CATEGORIES:                  # each letter in the alphabet
            path = os.path.join(tmp_path, category)  # create path to these images
            class_num = categorical_labels[CATEGORIES.index(category)]   # get one-hot ecoded label  

            for img in tqdm(os.listdir(path)):      # iterate over each image of hand signal
                try:
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # convert to array
                    img_array = cv2.resize(img_array, (160, 120))    # (width, height)
                    if direct_num < 5:                               # if not used for validation
                        training_set.append([img_array, class_num])  # add this to our training_data
                    elif direct_num == 5:                            # first validation data set
                        testing_set_1.append([img_array, class_num]) # add this to testing_data_1
                    else:                                            # second validation dat set
                        testing_set_2.append([img_array, class_num]) # add this to testing_data_2
                except Exception:  # in the interest in keeping the output clean...
                    print("hitting exception")
        
create_training_data()

# Shuffle and extract training data
random.shuffle(training_set)

X_train = []
y_train = []

for features, label in training_set:
    X_train.append(features)
    y_train.append(label)

# Changes X to a numpy array and palces pixel values in individual lists of one value contained within their own image lists
X_train = np.array(X_train).reshape(-1, 120, 160, 1)
y_train = np.array(y_train)


# Shuffle and extract testing data 1
random.shuffle(testing_set_1)

X_test_1 = []
y_test_1 = []

for features, label in testing_set_1:
    X_test_1.append(features)
    y_test_1.append(label)

X_test_1 = np.array(X_test_1).reshape(-1, 120, 160, 1)
y_test_1 = np.array(y_test_1)


# Shuffle and extract testing data 2
random.shuffle(testing_set_2)

X_test_2 = []
y_test_2 = []

for features, label in testing_set_2:
    X_test_2.append(features)
    y_test_2.append(label)

# Random test to show what images look like 
n = random.randint(0, 300)
test_array = X_test_2[n]
imgplot = plt.imshow(test_array)
plt.show()

X_test_2 = np.array(X_test_2).reshape(-1, 120, 160, 1)
y_test_2 = np.array(y_test_2)


import joblib

os.chdir(DATADIR) # Change current working directory

# Serialise datasets

# Pickle training data
pickle_out = open("X_train.pickle","wb")
joblib.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
joblib.dump(y_train, pickle_out)
pickle_out.close()

# Pickle testing data 1
pickle_out = open("X_test_1.pickle","wb")
joblib.dump(X_test_1, pickle_out)
pickle_out.close()

pickle_out = open("y_test_1.pickle","wb")
joblib.dump(y_test_1, pickle_out)
pickle_out.close()

# Pickle testing data 2
pickle_out = open("X_test_2.pickle","wb")
joblib.dump(X_test_2, pickle_out)
pickle_out.close()

pickle_out = open("y_test_2.pickle","wb")
joblib.dump(y_test_2, pickle_out)
pickle_out.close()
