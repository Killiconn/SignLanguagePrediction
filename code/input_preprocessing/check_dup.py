# Program to check for duplicates in frames fed to the CNN
# This program will be called by parent.py
# This program won't take care of irrelevant frames but will deal with duplicate frames instead
import numpy as np
import sys, os, cv2

def check_dup(frame_1, frame_2):
    ''' Take two numpy arrays and determine how close they are togther
    This could be done by calculating the euclidean distance between each value of each matrix
    i.e. the sum of ((frame_1 - frame_2)**2)
    '''
    assert frame_1.shape == frame_2.shape
    size = frame_1.shape[0] * frame_1.shape[1]
    dist = np.linalg.norm(frame_1-frame_2) / size
    return dist

def main():
    # Read in two frames for testing purposes
    DATADIR_1 = sys.argv[1]
    os.chdir(DATADIR_1)
    image = os.listdir()[0]
    img_array_1 = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    DATADIR_2 = sys.argv[2]
    os.chdir(DATADIR_2)
    image = os.listdir()[0]
    img_array_2 = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    check_dup(img_array_1, img_array_2)

if __name__ == '__main__':
    main()
