# This is a script to sort the dataset received from Alistair Sutherland (https://github.com/marlondcu/ISL/tree/master/Frames?fbclid=IwAR3921N7ApZR-X4eFDvYfylYFLjOYo5aCJvjnnVFHY5_92WMn_NwP-qx9SY) locally
# This script seperates the letters into individual directories
# Where these files will be located is taken off the commandline

import os, sys
import shutil
from tqdm import tqdm # shows a progress bar for an iteration while it's executing

# Different for Ubuntu so change line below
# "C:\\Users\\Bill\\3D Objects\\cnn_datasets\\ISL\\extracted_data_2" # Change this line Killian for your own laptop
# Take this off the commandline

# Location of images locally, letter directories will be created here
DATADIR = sys.argv[1]

# Letters in the dataset 
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Check if the directory already exists, and if not create it
for category in CATEGORIES:
    # make directory
    directory = os.path.join(DATADIR, category)
    if not os.path.exists(directory):
        # directory will be (current working directory + letter)
        os.mkdir(directory)

os.chdir(DATADIR) # Change current working directory to be the inputted directory

imgs_abs_path = os.listdir() # absolute path location of images currently

for img_path in tqdm(imgs_abs_path): # iterate through images
    # check the length of img_path
    # if it's of length one then it's one of the letter directories and ignore it, otherwise it's an image and process it and add it to a letter directory
    if len(img_path) > 1:
        # example img_path --> Person1-A-1-1.jpg
        # letter would equal "A" in this case
        letter = img_path.split("-")[1] 
        try:
            new_path = os.path.join(DATADIR, letter, img_path)
            old_path = os.path.join(DATADIR, img_path)
            os.rename(old_path, new_path)    # renaming paths
            shutil.move(old_path, new_path)  # moving file
            os.replace(old_path, new_path)   # get rid of old path
        except OSError:
            print("OSError occured on {:}".format(img_path))
