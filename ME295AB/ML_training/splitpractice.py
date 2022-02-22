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

## Work with split data

#####################################
def label_arrays(directory, row_dim):
	n = 0
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

shapehw = 192 #48
input_shape = (shapehw,shapehw,3)

base_dir = 'compiledtextures'
train_dir = os.path.join(base_dir,'train')
train_size = len(os.listdir(train_dir))
print('total {} classes: {}'.format(base_dir,train_size))

train_labels = label_arrays(train_dir, train_size)
train_num_images, train_size = train_labels.shape
print('Shape of train_labels array: {}'.format(train_labels.shape))

train_im_array_path = 'modelscompiled/texture_train_images_'+str(shapehw)+'_'+str(shapehw)+'.pt'
if exists(train_im_array_path):
	train_imgs = torch.load(train_im_array_path)
else:
	train_imgs = tensor_image_array(train_dir,input_shape,train_num_images)
	torch.save(train_imgs,train_im_array_path)
train_imgs = train_imgs.numpy()
print(train_imgs.shape)

## Reshuffle

randomize = np.arange(len(train_labels[:,1]))
np.random.shuffle(randomize)
train_labels = train_labels[randomize]
train_imgs = train_imgs[randomize,:,:,:]
print('Shuffled.')

#####################################

# We build the base model
base_model = VGG16(weights='imagenet',
	include_top=False,
	input_shape=input_shape)

# Freeze layers in base model so they do not train;
# want feature extractor to stay as before --> transfer learning
for layer in base_model.layers:
	#if layer.name == 'block5_conv1':
		#break	# Allow this layer to train
	layer.trainable = False
	print('Layer ' + layer.name + ' frozen.')

# We take the last layer of our the model and add it to our classifier
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(1000, activation='relu', name='fc1')(x) #1000
x = Dropout(0.30)(x) #0.3
x = Dense(train_size, activation='sigmoid', name='predictions')(x) #activation='softmax'
model = Model(base_model.input, x)

# We compile the model
model.compile(optimizer=SGD(lr=0.001), #lr=0.001
	loss='categorical_crossentropy',
	metrics=['accuracy'])

model.summary()

#####################################

# We start the training
epochs = 65
batch_size = 192#256
vsplit=0.35
# We train it
history = model.fit(train_imgs, train_labels,
	batch_size=batch_size,
	validation_split=vsplit,
	epochs=epochs)

timestr = time.strftime("%Y%m%d%H%M")
history_version = timestr+'_tl_vgg16_'+str(shapehw)+'inshape_'+str(epochs)+'epochs_'+str(batch_size)+'batch_'+str(vsplit)+'validationsplit'
model.save('modelscompiled/'+history_version+'.h5')

#####################################
'''
# Evaluate the accuracy and the loss in the test set
scores = model.evaluate(valid_imgs, valid_labels, verbose=1)
print("Scores: {}".format(scores))
'''
#####################################

# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.legend(["Accuracy", "Val_Accuracy"])
plt.title('Training and validation accuracy')
plt.savefig('modelscompiled/'+history_version+'_train_val_accuracy.jpg')

# Plot training and validation loss per epoch
plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.legend(["Loss", "Val_Loss"])
plt.title('Training and validation loss')
plt.savefig('modelscompiled/'+history_version+'_train_val_loss.jpg')

