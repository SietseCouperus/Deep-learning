# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:05:19 2021

@author: Sietse
"""

import pandas as pd
import seaborn as sb
import random
import os, os.path
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.regularizers import l2
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


with open("Y_final.txt", "rb") as fp:   # Unpickling
   Y = pickle.load(fp)

X = np.load("X_final.npy")
# print(X.shape)

x_train = X[:,:,0:30]
y_train = np.asarray(Y[0:30])
x_test = X[:,:,30:60]
y_test = np.asarray(Y[30:60])
# print(x_train.shape)
# print(x_test.shape)
# print(len(y_train))
# print(len(y_test))

def fix_input(data_old):
    data_new = np.empty((1,224,224,1))
    # print(data_new.shape)
    for i in range(data_old.shape[2]):
        x = data_old[:,:, i]
        # print(data.shape)
        x = x[np.newaxis, :, :, np.newaxis]
        # print(data.shape)
        data_new = np.concatenate([data_new, x], axis = 0)
    data_new = data_new[1:31, :, :, :]
    return data_new

X_train_new = fix_input(x_train)
X_test_new = fix_input(x_test)
# print(X_train_new.shape)
        
img = X_test_new[0,:,:,0]
img = Image.fromarray(img)
imshow(img, cmap=plt.cm.binary)
imshow(img, cmap=plt.get_cmap('gray'))
plt.show()   

# Two convolution layers of 64 kernels (size 3x3), 224x224 pixels
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', input_shape = (224, 224, 1), padding = 'same'))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
# Two convolution layers of 128 kernels (size 3x3), 112x112 pixels
model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
# Three convolution layers of 64 kernels (size 3x3), 56x56 pixels
model.add(Conv2D(256, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(256, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(256, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
# Three convolution layers of 64 kernels (size 3x3), 28x28 pixels
model.add(Conv2D(512, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(512, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(512, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
# Three convolution layers of 64 kernels (size 3x3), 14x14 pixels
model.add(Conv2D(512, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(512, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(512, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'))
#Flatten and continue as linear regression
model.add(Flatten())
model.add(Dense(4096, activation = 'relu', kernel_regularizer = l2(0.0005)))
model.add(Dropout(0.5))
model.add(Dense(4096, activation = 'relu', kernel_regularizer = l2(0.0005)))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

#specify hyperparameters of optimizer
SGD_M = SGD(learning_rate = 0.01, momentum = 0.9, nesterov = False, name = 'SGD_M')
RMSprop = RMSprop(learning_rate = 0.01, rho = 0.9, momentum = 0.9, epsilon = 10^-7, name = 'RMSprop')
#Specify loss function and optimizer to use
model.compile(loss = 'binary_crossentropy', optimizer = SGD_M, metrics = ['accuracy'])
#Specify batch size, number of epochs and what data to use
model.fit(x_train, y_train, batch_size = 1, epochs = 3, verbose = 2, validation_data = (x_test, y_test))


#Things to add:
    # Weight decay
    # Dropout
    # optimizers
    # Original VGG-16 model
    # Absolute value ReLU
    

model.summary()
# model_v.summary()
