# Program to decrease image size, so images can be processed more easily by the CNN
# Resizing images into a square withut stretching the images

#import matplotlib.pyplot as plt
import os
import cv2
import random
from add_noise import add_noise

# program will be fed an image and reduce it's size to only the required information

def reduce_image_size(img_array, max_dim=480):
    # change image size to (150, 150)
    
    # images have a black background i.e. all 0's in background
    # therfore the position of hand can be determined from the position of the first non zero pixel
    x, y, width, height = cv2.boundingRect(img_array)
    
    # for images which don't need padding as crop is ntot larger than the height of the image
    if max_dim < 480:
        # Centralise the image
        if (y + max_dim) > 480:
            y = (480 - max_dim) // 2
        
        if (x + max_dim) > 640:
            x = (640 - max_dim) // 2
        
        # Crop out noise
        img_array = img_array[y:y+max_dim, x:x+max_dim]
    
    # these images will need padding as the square dimensions will be graeter than the height of the original image
    else:
        # ratio of img_array is (480, 640)
        # centre the image and add padding

        # crop out unwanted info and keep the hand
        img_array = img_array[y:y+height, x:x+width]

        # figure out how much padding is needed to get to a square
        shape = img_array.shape 
        new_height, new_width = shape
        
        # now add padding to make a square image
        top, bottom, left, right = [10] * 4
        borderType = cv2.BORDER_CONSTANT

        # Finding biggest dimension
        # new_max_dim = new_width
        if new_height > new_width:
            # new_max_dim = new_height
            left = new_height - new_width

        else:
            top = new_width - new_height

        img_array = cv2.copyMakeBorder(img_array, top, bottom, left, right, borderType)

        # top and left most important for padding as hand in top left corner


    # Resize to reduce file size
    new_array = cv2.resize(img_array, (150, 150))

    # Add noise
    new_array = add_noise(new_array)
    
    #imgplot = plt.imshow(new_array)
    #plt.show()

    return new_array

def find_largest_dimension(img_array):
    # find largets dimension of matrix
    x, y, width, height = cv2.boundingRect(img_array)
    
    if width > height:
        return width
    
    else:
        return height

def main():
    # This is only ran locally when testing new features
    # Otherwise both functions are called in the dataset serialisation programs
    #DATADIR = "C:\\Users\\Bill\\3D Objects\\cnn_datasets\\ISL\\extracted_data_1"
    DATADIR = sys.argv[1]
    CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    # Show the program output for an image of each letter
    for letter in CATEGORIES:
        try:
            os.chdir(DATADIR + "\\" + letter)
            images = os.listdir()
            image = images[random.randint(0, 300)] # Random image
            img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            reduce_image_size(img_array, 560)

        except:
            print("An eroor occured")
        
    # Show image before
    #img = mpimg.imread(image)
    
    

if __name__ == "__main__":
    main()

