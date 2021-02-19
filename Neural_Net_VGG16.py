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
