#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from os.path import exists
import cv2
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import callbacks

from tensorflow.keras.layers import Dropout, Flatten, Dense

## Work with pre-split data


#####################################

shapehw = 32
input_shape = (shapehw,shapehw,3)




#####################################

# We build the base model
# base_model = VGG16(weights='imagenet',
#     include_top=False,
#     input_shape=input_shape)
# base_model.summary()




##



tic = time.time()
model = tf.keras.models.load_model('202111222114_tl_vgg16_0.55dropout_32inshape_0.00875LR_100epochs_384batch.h5')
toc = time.time()

model.summary()

print(toc-tic)


