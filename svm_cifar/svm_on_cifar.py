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

# The data must be downloaded once by running cifar10.maybe_download_and_extract()
# The data is around 168MB

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
class_names = cifar10.load_class_names()
class_names[0] = 'plane'
class_names[1] = 'auto'

#  The images to train, the labels (0-9) and the labels in 1hot [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
images_train, cls_train, labels_train = cifar10.load_training_data()

images_test, cls_test, labels_test = cifar10.load_test_data()

from cifar10 import img_size, num_channels, num_classes

#  Implémentation d'un SVM 1v9

def mySVM(images_train, cls_train, labels_train, images_test, cls_test, labels_test):
    # Data pre-processing to flatten the images
    # The training set used to be split 90-10 for validation when trying different algorithms.
    # Test set was only used to plot confusion matrices.
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
    
    # Pre-processing done


    # Toggle the comment between the next two lines for Linear 1vAll SVM or 1v1 RBF Kernel
    # clf = svm.LinearSVC()
    clf = svm.SVC(gamma='scale',decision_function_shape='ovo')


    # Train data
    clf.fit(train_data, train_class)

    # Prediction on test data to produce confusion matrix. Prediction was made with 10% of training data when trying various algorithms
    prediction = clf.predict(test_data)
    matrix = confusion_matrix(test_class,prediction)
    return matrix, clf

# Run the SVM here, the preprocessing is built into this function
matrix, clf = mySVM(images_train, cls_train, labels_train, images_test, cls_test, labels_test)

# Plot confusion matrix here
plt.figure()
plot_confusion_matrix(matrix,class_names)
plt.show()