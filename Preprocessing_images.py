# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:07:09 2021

@author: Sietse
"""

import matplotlib.pyplot as plt
import numpy as np

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io
import os, os.path
from PIL import Image, ImageOps


dir = 'C:/Users/Sietse/Documents/Biomolecular Sciences RUG/Deep Learning/Assignment 1; Tumour detection/Dataset/archive/'
list = os.listdir(dir + '/no') #get the images with or without a tumor
number_files = len(list)
print(number_files)
os.chdir(dir)
print(os.getcwd())

path = 'C:/Users/Sietse/Documents/Biomolecular Sciences RUG/Deep Learning/Assignment 1; Tumour detection/Dataset/archive/no/no'

for x in range(3):
    #retrieve every image in the directory
    ig = path + str(x) + ".jpg" 
    im = Image.open(ig)
    img_grayscale = ImageOps.grayscale(im) #convert from RGB to grayscale
    img_resized = img_grayscale.resize((224,224)) #resize to 224x224 for the VGG-16 net
    name = 'resizedN' + str(x)
    img_resized.save('pp_img/no/' + name + '.png') #save in a new directory as png files


