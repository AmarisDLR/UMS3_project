
import os
from os.path import exists
import cv2
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras import layers #import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import wandb
from wandb.keras import WandbCallback


#wandb.init(project="ums3_custom_mar15", entity="amarisdlr")

### Work with pre-split data

#####################################
def label_arrays(directory, row_dim):
    n = 0
    labels = []
    for root, subdir, files in os.walk(directory):
        for subfolder in subdir:
            classes = os.path.join(directory,subfolder)
            class_size = len(os.listdir(classes))
            temp_array = np.zeros((class_size,row_dim))
            temp_array[:,n] = 1
            if n >= 1:
                labels = np.concatenate((labels,temp_array))
            else:
                labels = temp_array
            n += 1
    return labels

def tensor_image_array(directory, input_shape,row_dim):
    array_imgs = torch.zeros(row_dim, input_shape[0],input_shape[1],input_shape[2])
    n = 0
    for root, subdir, files in os.walk(directory):
        for subim in files:
            subimage = root+"/"+subim
            img = cv2.imread(subimage)
            img_tensor = torch.from_numpy(img)/255.0
            img_tensor = img_tensor.resize_(input_shape)
            img_tensor = img_tensor.unsqueeze(dim=0)
            array_imgs[n,:,:,:] = img_tensor
            n += 1
    return array_imgs


#####################################

shapehw = 32
input_shape = (shapehw,shapehw,3)

base_dir = '../UMS3_dd_Mar15'
train_dir = os.path.join(base_dir,'train')
valid_dir = os.path.join(base_dir,'valid')

train_size = len(os.listdir(train_dir))
valid_size = len(os.listdir(train_dir))
if train_size == valid_size:
    print('total {} classes: {}'.format(base_dir,train_size))

train_labels = label_arrays(train_dir, train_size)
train_num_images, train_size = train_labels.shape
print('Shape of train_labels array: {}'.format(train_labels.shape))

valid_labels = label_arrays(valid_dir, valid_size)
valid_num_images, valid_size = valid_labels.shape
print('Shape of valid_labels array: {}'.format(valid_labels.shape))

train_im_array_path = base_dir+'_train_images_'+str(shapehw)+'_'+str(shapehw)+'.pt'
if exists(train_im_array_path):
    train_imgs = torch.load(train_im_array_path)
else:
    train_imgs = tensor_image_array(train_dir,input_shape,train_num_images)
    torch.save(train_imgs,train_im_array_path)
train_imgs = train_imgs.numpy()
print(train_imgs.shape)

valid_im_array_path = base_dir+'_valid_images_'+str(shapehw)+'_'+str(shapehw)+'.pt'
if exists(valid_im_array_path):
    valid_imgs = torch.load(valid_im_array_path)
else:
    valid_imgs = tensor_image_array(valid_dir,input_shape,valid_num_images)
    torch.save(valid_imgs,valid_im_array_path)
valid_imgs = valid_imgs.numpy()
print(valid_imgs.shape)

### Reshuffle train and validation images

randomize_train = np.arange(len(train_labels[:,1]))
np.random.shuffle(randomize_train)
train_labels = train_labels[randomize_train]
train_imgs = train_imgs[randomize_train,:,:,:]

randomize_valid = np.arange(len(valid_labels[:,1]))
np.random.shuffle(randomize_valid)
valid_labels = valid_labels[randomize_valid]
valid_imgs = valid_imgs[randomize_valid,:,:,:]

print('\n\nShuffled.\n\n')

#####################################

### Build the base model

lr = 0.0001
model = models.Sequential()
model.add(layers.experimental.preprocessing.Resizing(128, 128, interpolation="gaussian", input_shape=input_shape))
model.add(layers.Conv2D(96, 5, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(128, 5, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(128, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(128, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.65))
model.add(layers.Dense(350, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(train_size, activation='softmax'))




model.summary()

opt = RMSprop(lr = lr) # stochastic gradient descent method with learning rate lr = 0.01    
model.compile(optimizer=opt, 
    loss='categorical_crossentropy',
    metrics=['accuracy','categorical_accuracy','categorical_crossentropy'])

#####################################

### Start the training
epochs = 75
batch_size = 256

'''wandb.config = {
  "learning_rate": lr,
  "epochs": epochs,
  "batch_size": batch_size,
  "optimizer": opt
}'''

# Start training
start_train = time.time()
history = model.fit(train_imgs, train_labels,
    batch_size=batch_size,
    validation_data=(valid_imgs, valid_labels),
    epochs=epochs,
    callbacks=[WandbCallback()])
end_train = time.time()
training_time = (end_train-start_train)/60

timestr = time.strftime("%Y%m%d%H%M")
history_version = timestr

savemodel = input("Save model? y/n")

if savemodel == 'y' or savemodel == "Y":
	model.save('models/'+history_version+'.h5')
	print("Model saved.")
else:
	pass
print("End training")
	

