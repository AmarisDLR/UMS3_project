import os
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,Ftrl, Adamax,Nadam, RMSprop, SGD, Adagrad, schedules
from tensorflow.keras.callbacks import ModelCheckpoint

import wandb,sys
from wandb.keras import WandbCallback


defaults = dict(
    activation_type='softmax',
    dense_units=256,#3256
    dropout=0.15,
    epochs = 150,
    learn_rate=0.00001, #0.000020
    batch_size=25,
    shapehw = 96,
    iav=0.21,
    eps=1.05e-7,
    train_steps = 6500,
    valid_steps = 3500,
    zoomr =(0.8,1.1),
    brightnessr = (0.8,1.1),
    rotationr = 5
    )

resume = sys.argv[-1] == "--resume"
wandb.init(config=defaults,project="ums3_imdatagen_aprx", resume=resume)
config = wandb.config
run_name = wandb.run.name

base_dir = '../UMS3_dd_April10/training_randomized'
train_dir = os.path.join(base_dir,'train')
valid_dir = os.path.join(base_dir,'val')
test_dir = os.path.join(base_dir,'test')


train_datagen = ImageDataGenerator(rescale = 1./255,
                                      rotation_range=config.rotationr,
                                      zoom_range=config.zoomr,
                                      brightness_range=config.brightnessr)


valid_datagen = ImageDataGenerator(rescale = 1./255)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(config.shapehw,config.shapehw),
                                                    class_mode='categorical',
                                                    batch_size=config.batch_size,
                                                    shuffle=True)
train_classes = train_generator.num_classes
train_size = np.shape(train_generator.filenames)[0]

valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    target_size=(config.shapehw,config.shapehw),
                                                    class_mode='categorical',
                                                    batch_size=config.batch_size,
                                                    shuffle=True)

valid_size = np.shape(valid_generator.filenames)[0]



generated_image = train_generator[0][0][0]
plt.imshow(generated_image)

base_model =VGG16(weights='imagenet',
    include_top=False,
    input_shape=(config.shapehw,config.shapehw,3))


for layer in base_model.layers:
    layer.trainable=True
    
flat1 = Flatten()(base_model.layers[-1].output)
class1 = Dense(config.dense_units, activation='relu')(flat1)
class1 = Dropout(config.dropout)(class1) #0.3
output = Dense(train_classes, activation=config.activation_type)(class1)

model = Model(inputs = base_model.inputs, outputs = output)

# Compile the model

optimizer=Nadam(learning_rate=config.learn_rate,
                  epsilon=config.eps)

model.compile(optimizer=optimizer, 
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.summary()

model_name = run_name
saved_model = os.path.join('models',model_name)
checkpoint =  ModelCheckpoint(filepath=saved_model,verbose=1,save_best_only=True,
                              monitor='val_accuracy',
                              mode='max')

history = model.fit(
        train_generator,
        steps_per_epoch= config.train_steps//train_size,
        validation_data=valid_generator,
        validation_steps=config.valid_steps//valid_size,
        epochs = config.epochs,
        callbacks=[WandbCallback(),checkpoint])


model_name2 = run_name+'_weights'
saved_model2 = os.path.join('models',model_name2)
model.save_weights(saved_model2)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']
  
epochs = range(1, len(loss_values) + 1)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
#
# Plot the model accuracy vs Epochs
#
ax[0].plot(epochs, accuracy, color='blue', label='Training accuracy')
ax[0].plot(epochs, val_accuracy, color='slategrey', linestyle='--', label='Validation accuracy')
ax[0].set_title('Training & Validation Accuracy', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Accuracy', fontsize=16)
ax[0].legend()

#
# Plot the loss vs Epochs
#
ax[1].plot(epochs, loss_values, color='blue', label='Training loss')
ax[1].plot(epochs, val_loss_values, color='slategrey', linestyle='--', label='Validation loss')
ax[1].set_title('Training & Validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].legend()

fig.savefig('models/'+model_name+'_training_validation.png')

#
# Evaluate the model accuracy and loss on the test dataset
#

test_generator = valid_datagen.flow_from_directory(test_dir,
                                                    target_size=(config.shapehw,config.shapehw),
                                                    class_mode='categorical',
                                                    batch_size=2,
                                                    shuffle=False)

test_size = np.shape(valid_generator.filenames)[0]

score = model.evaluate(test_generator,batch_size=3,verbose=1)
#
# Print the loss and accuracy
#
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(run_name)

'''from tensorflow.keras.preprocessing import image 

subimg= "C:/Users/amari/UMS3_project/ME295AB/UMS3_dd_April10/training_randomized/test/stringing_low/202111021552_UMS3_random33_starpuzzle_5infill_zigzag_24597img_jpg.rf.a8bad12e7f2b8af190d69c45642c9196.jpg"


img = image.load_img(subimg, target_size=(config.shapehw,config.shapehw,3))
plt.figure()
plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

print(train_generator.class_indices)
classx = model.predict(x)
print(classx)
print('bloop')

# run model inference on test data set
test_predict_array = []
label_array_test = []
n= 1


print(len(test_generator))
for t in test_generator:
    label_array_test.append(t[1])
    test_predict_array.append(model.predict(t[0]))	
    n+=1
    if n == 500000:
        print('break')
        break

test_predict_array = np.concatenate(test_predict_array, axis = 0)
label_array_test = np.concatenate(label_array_test, axis = 0)


# create ROC curves
print('calc ROC curve')
fpr_test, tpr_test, threshold_test = roc_curve(label_array_test[:,1], test_predict_array[:,1])
test_auc = auc(fpr_test, tpr_test)*100
print(f'Test AUC = {test_auc}')
# plot ROC curves
plt.figure()
plt.plot(tpr_test, fpr_test, lw=2.5, label="Test, AUC = {:.1f}%".format(test_auc))
plt.xlabel(r'True positive rate')
plt.ylabel(r'False positive rate')
plt.semilogy()
plt.ylim(0.001, 1)
plt.xlim(0, 1)
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
plt.savefig('models/'+model_name+'_auc_curve.jpg')

print('done.')'''