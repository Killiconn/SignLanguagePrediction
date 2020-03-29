import numpy as np
import os
import cv2
from tqdm import tqdm # shows a progress bar for an iteration while it's executing
import random
import sys
import random

# Program to change images by adding noise
# Gonna do this by looking for blackness and settiing to a random colour between zero and one
# This should make the neural network generalise more as all images won't have the same black background


def add_noise(img_array):
    # Add random grey background to images to replace black background already in images
    img_array[img_array < 25] = random.randint(0, 255)
    return img_array


def main():
    # This is just for testing image. this main only gets called during testing
    DATADIR = "C:\\Users\\Bill\\3D Objects\\cnn_datasets\\ISL\\extracted_data_1"
    CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y"]
    # Show the program output for an image of each letter
    for letter in CATEGORIES:
        try:
            os.chdir(DATADIR + "\\" + letter)
            images = os.listdir()
            image = images[random.randint(0, 300)] # Random image
            img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            img_array = reduce_image_size(img_array)
            img_array = add_noise(img_array)
            img_plot = plt.imshow(img_array, cmap='gray')
            plt.show()

        except:
            print("An error occured")

if __name__ == '__main__':
    main()
