import pandas as pd
import seaborn as sb
import numpy as np
import random
import cv2
import os, os.path
import tensorflow as tf
from tensorflow import keras
import pickle

from PIL import Image, ImageOps
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img

#--------------------------------------------------------------# PREPROCESSING N1 ONLY RUN ONCE

# For positive files
print("Yes")
# Counting how many images there with label yes 
dir = "DL/label1/"
list = os.listdir(dir)
number_files = len(list)
print(number_files)

path = "DL/label1/y"

for x in range(number_files):
    ig = path + str(x) + ".jpg"
    im = Image.open(ig)
    img_grayscale = ImageOps.grayscale(im)
    img_resized = img_grayscale.resize((224,224))
    name = 'resizedY' + str(x)
    img_resized.save('YesGS/' + name + '.png')

# For negative files
print(np.__version__)
print("No")
# Counting how many images there with label no 
dir2 = "DL/label2/"
list2 = os.listdir(dir2)
number_files2 = len(list2)
print(number_files2)

path = "DL/label2/no"

for x in range(number_files2):
    ig = path + str(x) + ".jpg"
    im = Image.open(ig)
    img_grayscale = ImageOps.grayscale(im)
    img_resized = img_grayscale.resize((224,224))
    name = 'resizedY' + str(x)
    img_resized.save('NoGS/' + name + '.png')

# Counting how many images there with label no 
dir = "noGS"

for img_id in range(split):
    img = load_img(dir + "/" + name + '.png')
    pass
"""
"""
#--------------------------------------------------------------# NUMPY ARRAY TRAIN X YES
# Counting how many images there with label yes 
dir = "yesGS"
list = os.listdir(dir)
number_files = len(list)

split = 1
split = split * number_files
split = int(split)
print("Split: ", split)

#TRAIN YES
for img_id in range(split):
    name = 'resizedY' + str(img_id)
    img = load_img(dir + "/" + name + '.png')
    img = ImageOps.grayscale(img)
    if img_id == 0:
        arr = img_to_array(img)
    else:
        f_arr = img_to_array(img)
        arr = np.concatenate([arr, f_arr], axis= 2)

#--------------------------------------------------------------# NUMPY ARRAY TRAIN X NO 
dir = "noGS"

for img_id in range(split):
    name = 'resizedY' + str(img_id)
    img = load_img(dir + "/" + name + '.png')
    img = ImageOps.grayscale(img)
    f_arr = img_to_array(img)
    arr = np.concatenate([arr, f_arr], axis= 2)

np.save("Non_shuffled_x.npy",arr)
X = np.load("Non_shuffled_x.npy")

#--------------------------------------------------------------# LIST TRAIN Y (BOTH)
listofzeros = [0] * split
listofones = [1] * split
Y = listofones + listofzeros

#--------------------------------------------------------------# SHUFFLING LIST AND ARRAY
proportion = 0.8

listReg = np.arange(0, split*2, 1).tolist()

np.random.seed(420)
np.random.shuffle(listReg)
#print(listReg)

Y_final = []
i = 0

for x in listReg:
    Y_final.append(Y[x])
    i += 1
    if i <= 1:
        X_final = X[:,:,x]
        X_final = X_final[:, :, np.newaxis]
    else:
        arr = X[:,:,x]
        arr = arr[:,:, np.newaxis]
        X_final = np.concatenate([X_final, arr], axis= 2)

np.save("X_final.npy",X_final)


with open("Y_final.txt", "wb") as fp:   #Pickling
    pickle.dump(Y_final, fp)
"""
with open("Y_final.txt", "rb") as fp:   # Unpickling
   Y_final = pickle.load(fp)

print(Y_final)
#print(X_final.shape)
#--------------------------------------------------------------# DISPLAY THE IMAGE FROM NP ARRAY
"""
img = X_final[:,:,0]
img = Image.fromarray(img)
imshow(img, cmap=plt.cm.binary)
imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
"""
#--------------------------------------------------------------# SPLIT TRAIN AND TEST



#--------------------------------------------------------------# NUMPY ARRAY TEST
"""
#TEST
dir = "yesGS"
list = os.listdir(dir)
number_files = len(list)

#TEST YES
for img_id in range(split, number_files):
    name = 'resizedY' + str(img_id)
    img = load_img(dir + "/" + name + '.png')
    img = ImageOps.grayscale(img)
    if img_id == 0:
        arr = img_to_array(img)
        print(arr.shape)
        print(arr)
    else:
        f_arr = img_to_array(img)
        arr = np.concatenate([arr, f_arr], axis= 2)

#TEST NO 
dir = "noGS"

for img_id in range(split, number_files):
    name = 'resizedY' + str(img_id)
    img = load_img(dir + "/" + name + '.png')
    img = ImageOps.grayscale(img)
    f_arr = img_to_array(img)
    arr = np.concatenate([arr, f_arr], axis= 2)

print(arr.shape)
np.save("Test_X.npy",arr)
"""

#--------------------------------------------------------------# USELESS

# not to be used - for some funky colors 
def rgb2gray_approx(rgb_img):
    
    Convert *linear* RGB values to *linear* grayscale values.
    
    red = rgb_img[:, :, 0]
    green = rgb_img[:, :, 1]
    blue = rgb_img[:, :, 2]

    gray_img = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue)

    gray_img = (
         red * 0
        +  green * 0
        +  blue * 1.5)

    return gray_img
"""
