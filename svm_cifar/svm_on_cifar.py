import cifar10
import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
import math
import os
import matplotlib.pyplot as plt
import time
import itertools
import matplotlib.colors as colors


#Draws 9 images in a 3x3 and their class
def plot_images(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
            
        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=truncate_colormap(plt.cm.OrRd, 0.1, 0.7)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#  The 10 classes in text
start = time.time()
class_names = cifar10.load_class_names()
class_names[0] = 'plane'
class_names[1] = 'auto'

#  The images to train, the labels (0-9) and the labels in 1hot [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
images_train, cls_train, labels_train = cifar10.load_training_data()

images_test, cls_test, labels_test = cifar10.load_test_data()

from cifar10 import img_size, num_channels, num_classes

print("Importation des données complété en {0:4.2f} secondes.".format(time.time()-start))
#  Implémentation d'un SVM 1v9

def mySVM(images_train, cls_train, labels_train, images_test, cls_test, labels_test):
    nbImg = int(len(images_train))
    nbTest = int(len(images_test))
    start = time.time()
    
    train_data = [[None]*len(images_train[0])*len(images_train[0][0])*len(images_train[0][0][0]) for l in range(nbImg)]
    test_data = [[None]*len(images_test[0])*len(images_test[0][0])*len(images_test[0][0][0]) for l in range(nbTest)]
    
    train_class = [0 for l in range(nbImg)]
    test_class = [0 for l in range(nbTest)]

    for i in range(nbImg):
        train_data[i] = images_train[i].flatten()
        train_class[i] = cls_train[i]
    for i in range(nbTest):
        test_data[i] = images_test[i].flatten()
        test_class[i] = cls_test[i]

    train_class = np.asarray(train_class)
    test_class = np.asarray(test_class)
    print("Pre-process des données complété en {0:4.2f} secondes.".format(time.time()-start))
    start = time.time()
    #clf = svm.LinearSVC()
    clf = svm.SVC(gamma='scale',decision_function_shape='ovo')
    clf.fit(train_data, train_class)
    print("Training complété en {0:.0f} secondes.".format(time.time()-start))
    start = time.time()
    prediction = clf.predict(test_data)
    print("Prédiction complétée en {0:.0f} secondes.".format(time.time()-start))
    matrix = confusion_matrix(test_class,prediction)
    return matrix, clf


#images = images_train[6:15]
#cls_true = cls_train[6:15]
#plot_images(images, cls_true, smooth=True)

#  Implémentation d'un SVM 1 vs 9
matrix, clf = mySVM(images_train, cls_train, labels_train, images_test, cls_test, labels_test)

# matrix = [[316, 113,  37, 188,  65,  20,  16,  25, 176,  44],
#        [ 33, 561,  25,  91,  55,  26,  38,  24,  68,  79],
#        [ 83,  66, 155, 255, 201,  44,  99,  32,  48,  17],
#        [ 24,  89,  62, 441, 115,  91,  91,  27,  32,  28],
#        [ 34,  48,  88, 210, 365,  51, 112,  48,  27,  17],
#        [ 25,  73,  49, 375, 157, 158,  56,  33,  54,  20],
#        [ 10,  62,  61, 240, 156,  45, 361,  21,  27,  17],
#        [ 34,  81,  54, 196, 155,  74,  37, 276,  53,  40],
#        [ 86, 160,  21, 112,  49,  15,   9,   7, 491,  50],
#        [ 51, 342,  36,  81,  54,  22,  37,  40,  97, 240]]
# matrix = np.asarray(matrix)
print("{0:.0f} éléments dans la matrice de confusion. {1:.0f} éléments sur la diagonale.".format(np.sum(matrix),np.trace(matrix)))

plt.figure()
plot_confusion_matrix(matrix,class_names)
plt.show()

# print("hello world")