import os

import numpy as np
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299 * 255, 0.587 * 255, 0.114 * 255])

list_dirs = os.walk("intween")
for root, dirs, files in list_dirs:
    for d in dirs:
        print("dir: " + os.path.join(root, d))
    for f in files:
        # TODO
        print("file: " + os.path.join(root, f))
        imageName = os.path.join(root, f)
        imageOrigin = mpimg.imread(imageName)
        imageGray = rgb2gray(imageOrigin)
        imageArray = imageGray.astype("uint8")
        matrix = [[[imageArray[row][col]] for col in range(28)] for row in range(28)]
        print(imageArray)


