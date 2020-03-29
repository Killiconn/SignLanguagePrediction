# Program to resize the (3000, 3000) images to (150, 150) for image segmentation and changing the annotations

import sys, os

# "C:\\Users\\Bill\\3D Objects\\cnn_datasets\\hand_recognition_data\\ISL_test"
DATADIR = sys.argv[1]
os.chdir(DATADIR)

import numpy as np
from PIL import Image
import os
from tqdm import tqdm


for image in tqdm(os.listdir(DATADIR)):
    if image.strip() != "resized_images":
        outfile = os.path.join(DATADIR, "..", "resized_images", image)
        path = os.path.join(DATADIR, image)
        img = Image.open(path)
        #img = img.rotate(180, Image.NEAREST, expand = 1)
        img.thumbnail((150, 150), Image.ANTIALIAS)
        img.save(outfile, "JPEG")



