import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, matthews_corrcoef,brier_score_loss
import itertools

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from tensorflow.keras.models import load_model

def class_index(actual_array):
    index = np.nonzero(actual_array)
    index = np.transpose(index)
    return index[0][0]

def auc_curves(generator):
    test_predict_array = []
    test_actual_array = []

    for i in range(len(generator)):
        img, label = generator.next()
        for j in range(len(img[:])):
            label_check = label[j]
            test_actual_array.append(label_check)
            img_check = img[j]
            x = img_to_array(img_check)
            x = np.expand_dims(x, axis=0)
            
            classx = model.predict(x)
            test_predict_array.append(classx)
            
    test_predict_array = np.concatenate(test_predict_array, axis = 0)

    np.shape(test_actual_array)[0]
    temp_predict = []
    temp_actual = []
    temp_test_auc = []
    
    plt.figure(0,facecolor='white')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel(r'True positive rate')
    plt.ylabel(r'False positive rate')
    plt.grid(True)    
    
    plt.figure(1,facecolor='white')
    plt.xlim(0.7, 1.015)
    plt.ylim(-0.015, 0.3)
    plt.xlabel(r'True positive rate')
    plt.ylabel(r'False positive rate')
    plt.grid(True)

    for i in range(generator.num_classes):
        check_arr = np.zeros(7)
        check_arr[i]=1
        check_arr_index = class_index(check_arr)
        for j in range(np.shape(test_actual_array)[0]):
            test_actual_index = class_index(test_actual_array[j])
            if check_arr_index == test_actual_index:
                temp_predict.append(test_predict_array[j])
                temp_actual.append(test_actual_array[j])

        temp_actual = np.concatenate(temp_actual, axis = 0)
        temp_predict = np.concatenate(temp_predict, axis = 0)       
        fpr_test, tpr_test, threshold_test = roc_curve(temp_actual, temp_predict)
        test_auci = auc(fpr_test, tpr_test)*100
        plt.figure(0)
        plt.plot(tpr_test, fpr_test, lw=1.0, label="Class {:d}, AUC = {:.1f}%".format(i,test_auci))
        plt.figure(1)
        plt.plot(tpr_test, fpr_test, lw=1.0, label="Class {:d}, AUC = {:.1f}%".format(i,test_auci))
        temp_test_auc.append(test_auci)
        temp_predict = []
        temp_actual = []
        pass

    plt.figure(0)
    plt.title('ROC Curve Analysis', fontsize=16)
    plt.legend(loc='upper left')
  
    plt.figure(1)
    plt.title('ROC Curve Analysis', fontsize=16)
    plt.legend(loc='upper left')
    
    plt.show()

    pass

def conf_matrix(y_true, y_pred, classes,normalize,title):
    plt.figure(facecolor='white')
    cm = confusion_matrix(y_true, y_pred)
    cm_normalize = cm/cm.astype(np.float).sum(axis=1)
    # Configure Confusion Matrix Plot Aesthetics (no text yet) 
    plt.imshow(cm_normalize, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.colorbar()
    
    # Place Numbers as Text on Confusion Matrix Plot
    # thresh = cm_normalize.max() / 2.
    for i, j in itertools.product(range(cm_normalize.shape[0]), range(cm_normalize.shape[1])):
        plt.text(j, i, format(cm_normalize[i, j],'.2f' if cm[i, j] > 0 else '.2g'),
                 horizontalalignment="center",
                 color="white" if cm_normalize[i, j] > .6 else "steelblue",
                 fontsize=12)
    # Plot
    plt.tight_layout()
    plt.show()
    
def plot_classification_report(classification_report, number_of_classes, title, cmap):
    plt.figure(facecolor='white')
    ax = plt.gca()
    ax.set_axis_off()
    plt.title('Classifiction Report', fontsize=24)

    rpt = classification_report.replace('avg / total', '      avg')
    rpt = classification_report.replace('support', 'N Obs')

    plt.annotate(rpt, 
                 xy = (.5,.65), 
                 xytext = (150, -75), 
                 xycoords='axes fraction', textcoords='offset points',
                 fontsize=20, ha='right',va='center')
    plt.show()  
    print(rpt)

    
    lines = classification_report.split('\n')
    #drop initial lines
    lines = lines[2:2+number_of_classes]

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines:
        t = list(filter(None, line.strip().split('  ')))
        if len(t) < 4: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append('Class '+str(t[0]))
        plotMat.append(v)
        
    plt.figure(facecolor='white')
    plt.gca().set(frame_on=True)
    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = class_names
    sns.heatmap(np.array(plotMat),yticklabels=yticklabels,xticklabels=xticklabels,
                cmap=cmap,annot=True,)
    ax = plt.gca()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")
    plt.xlabel(xlabel,fontsize=14)
    plt.ylabel(ylabel,fontsize=14)
    plt.title(title,fontsize=18)
    plt.show()
    
    

base_dir = '../UMS3_dd_April10/training_randomized'
train_dir = os.path.join(base_dir,'train')
valid_dir = os.path.join(base_dir,'val')
test_dir = os.path.join(base_dir,'test')


datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=3,
                             brightness_range=[0.35,1.35],
                             zoom_range=[0.85,1.25],
                             channel_shift_range=1.25)


c = 'models/charmed-silence-151'
model = load_model(c)
model_shape=model.layers[0].input.get_shape().as_list()
shapehw = model_shape[1]

test_generator = datagen.flow_from_directory(test_dir,
                                             target_size=(shapehw,shapehw),
                                             class_mode='categorical',
                                             batch_size=15,
                                             shuffle=False)

subimg= "C:/Users/amari/Downloads/Picture4.jpg"


img = load_img(subimg, target_size=(shapehw,shapehw,3))
plt.figure()
plt.imshow(img)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

print(test_generator.class_indices)
classx = model.predict(x)

print(classx)
'''

# Plot AUC curves
auc_curves(test_generator)



true_classes = test_generator.classes
test_preds = model.predict(test_generator)
test_preds_classes = np.argmax(test_preds, axis = 1)
class_names = test_generator.class_indices.keys()
print(class_names)


# Plot Confusion Matrix
class_labels = ["Class 0", "Class 1","Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]
conf_matrix(true_classes, test_preds_classes, 
                         classes= class_labels,
                         normalize=True, 
                         title='Confusion Matrix')
    
# Plot Classification Report
rpt = classification_report(true_classes, test_preds_classes)
plot_classification_report(rpt,test_generator.num_classes,'Classification Report',plt.cm.Blues)

mcc = matthews_corrcoef(true_classes, test_preds_classes)
print(mcc)
#A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. 

bsll = brier_score_loss(true_classes, test_preds_classes)
print(bsll)'''