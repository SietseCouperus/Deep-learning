# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:05:19 2021

@author: Sietse
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.vgg16 import VGG16

model = Sequential()
model_v = VGG16()

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
model.add(Dense(4096, activation = 'relu'))
model.add(Dense(4096, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#Things to add:
    # Weight decay
    # Dropout
    # optimizers
    # Original VGG-16 model
    # Absolute value ReLU
    

model.summary()
# model_v.summary()