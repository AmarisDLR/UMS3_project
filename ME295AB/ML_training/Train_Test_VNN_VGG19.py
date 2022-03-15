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
from tensorflow.keras.applications import VGG19
from tensorflow.keras import callbacks

from tensorflow.keras.layers import Dropout, Flatten, Dense

import wandb
from wandb.keras import WandbCallback


wandb.init(project="ums3_vgg19", entity="amarisdlr")

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

shapehw = 64
input_shape = (shapehw,shapehw,3)

base_dir = '../UMS3_dd_Mar08'
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

## Reshuffle

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

# We build the base model
base_model = VGG19(weights='imagenet',
    include_top=False,
    input_shape=input_shape)
base_model.summary()

# Freeze layers in base model so they do not train;
# want feature extractor to stay as before --> transfer learning
for layer in base_model.layers:
    '''if layer.name == 'block4_conv1':
        break # Allow this layer to train
    if layer.name == 'block5_conv1':
        break # Allow this layer to train'''
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen.')

# We take the last layer of our the model and add it to our classifier
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(16, activation='relu', name='fc1')(x) #800 #relu
dropout=0.15
x = Dropout(dropout)(x) #0.3
x = Dense(train_size, activation='softmax', name='predictions')(x)#softmax

model = Model(base_model.input, x)

# We compile the model
lr = 0.00001
model.compile(optimizer=RMSprop(lr=lr), #lr=0.001
    loss='categorical_crossentropy',
    metrics=['accuracy','categorical_accuracy','categorical_crossentropy'])

model.summary()

#####################################

# We start the training
epochs = 75
batch_size = 256

wandb.config = {
  "learning_rate": lr,
  "epochs": epochs,
  "batch_size": batch_size,
  "dropout_rate":dropout
}

# We train it
start_train = time.time()
history = model.fit(train_imgs, train_labels,
    batch_size=batch_size,
    validation_data=(valid_imgs, valid_labels),
    epochs=epochs,
    callbacks=[WandbCallback()])
end_train = time.time()
training_time = (end_train-start_train)/60

timestr = time.strftime("%Y%m%d%H%M")
history_version = timestr+'_tl_vgg16_'+str(dropout)+'dropout_'+str(shapehw)+'inshape_'+str(lr)+'LR_'+str(epochs)+'epochs_'+str(batch_size)+'batch'
model.save('models/'+history_version+'.h5')

#####################################

# Evaluate the accuracy and the loss in the test set
scores = model.evaluate(valid_imgs, valid_labels, verbose=1)
print("Scores: {}".format(scores))
scores0 = "%.2f" % scores[0]
scores1 = "%.2f" % scores[1]

#####################################

# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']


subimg= valid_dir+'/stringing/1634702965_UMS3_hexagon_squareoval_15infill_lines_9911img_jpg.rf.44e8c67f064a0d4c792dcd9860d5cb90.jpg'
img = cv2.imread(subimg)
img_tensor = torch.from_numpy(img)/255.0
img_tensor = img_tensor.resize_(input_shape)
img_tensor = img_tensor.unsqueeze(dim=0)
img_tensor = img_tensor.numpy()

prediction = model.predict(img_tensor)
print('Actually: overextrusion; Prediction: {}'.format(prediction))

savefigs = 'y'# input('Save figs? (y/n) ')

if savefigs=='y':
	# Get number of epochs
	epochs = range(len(acc))

	# Plot training and validation accuracy per epoch
	plt.plot(epochs, acc)
	plt.plot(epochs, val_acc)
	plt.legend(["Accuracy", "Val_Accuracy"])
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.title('Training and validation accuracy: '+'Val Loss='+scores0 +' Val Accuracy='+scores1)
	plt.savefig('models/'+history_version+'_train_val_accuracy.jpg')

	# Plot training and validation loss per epoch
	plt.figure()
	plt.plot(epochs, loss)
	plt.plot(epochs, val_loss)
	plt.legend(["Loss", "Val_Loss"])
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.title('Training and validation loss: '+'Val Loss='+scores0+' Val Accuracy='+scores1)
	plt.savefig('models/'+history_version+'_train_val_loss.jpg')
	print('Plots saved.')
else:
	print('No plots saved.\n')
print('Program completed.\n')
print("Training took: "+str(training_time)+" min.")


'''
# In[ ]:


for root, subdir, files in os.walk(train_dir):
    for subfolder in subdir:
        print(subfolder)


# In[1]:


import os
from os.path import exists
import cv2
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

import tensorflow as tf


from tensorflow import keras
tic = time.time()
model = keras.models.load_model('UMS3_0/202111222308_tl_vgg16_0.55dropout_32inshape_0.0001LR_75epochs_256batch.h5')
toc = time.time()
toc-tic


# In[20]:


shapehw = 32
input_shape = (shapehw,shapehw,3)

base_dir = 'UMS3_0'
train_dir = os.path.join(base_dir,'train')
valid_dir = os.path.join(base_dir,'valid')
subimg= valid_dir+'/stringy/1634329033_34638img_jpg.rf.f2c682789e67e01b3b663f2c2329892b.jpg'


# In[26]:


subimg= '../../UMS3_project/ME295AB/split34_gcodeoverlay.jpg'
shapehw = 32
input_shape = (shapehw,shapehw,3)
img = cv2.imread(subimg)
img_tensor = torch.from_numpy(img)
img_tensor = img_tensor.resize_(input_shape)/255
img_tensor = img_tensor.unsqueeze(dim=0)
img_tensor = img_tensor.numpy()

prediction = model.predict(img_tensor)
prediction


# In[14]:


classes = list(np.argmax(prediction, axis=1))
classes


# In[ ]:


'''

