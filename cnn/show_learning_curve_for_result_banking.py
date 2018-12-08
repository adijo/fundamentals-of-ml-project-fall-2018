#Small utility to visualize the learning curves.
import re
import sys
import glob

filelist = []
if len(sys.argv) == 1:
    filelist = glob.glob("results\\banking\\*.txt")
else:
    #Also accepts wildcards
    for log_filename in sys.argv[1:]:
        filelist += glob.glob(log_filename)

print(filelist)
#Call the script using "python show_learning_curve_for_result.py results/2018-11-25-12_51_21.txt"

regex = re.compile("Epoch \[(\d+)/(\d+)\], Step \[(\d+)/(\d+)\], Loss: (\d+\.\d+), Accuracy:(\d+\.\d+)")
regex2 = re.compile("lr: (\d+\.\d+)") #For the learning rate
regex3 = re.compile("Epoch (\d+)\s+validation accuracy=\s+(\d+\.\d+)") #For the validation accuracy for each epoch

max_epoch = 0
loss_dict = {}
accuracy_dict = {}
epoch_dict = {}
learning_rate_dict = {}
validation_accuracy_dict={}

import matplotlib.pyplot as plt

print("Parsing", filelist)
for log_filename in filelist:
    with open(log_filename) as f:
      for line in f:
        result = regex.search(line)
        if result:
            epoch = result[1]
            total_epoch = result[2]
            step = result[3]
            total_step = result[4]
            loss = float(result[5])
            accuracy  = float(result[6])
            print(epoch, total_epoch, step, total_step, loss, accuracy)
            if(int(total_epoch)>max_epoch): max_epoch=int(total_epoch)
            if step == total_step:
                if not log_filename in loss_dict: #Create the arrays when they are needed
                    loss_dict[log_filename]=[]
                    epoch_dict[log_filename]=[]
                    accuracy_dict[log_filename]=[]
                    validation_accuracy_dict[log_filename]=[]
                loss_dict[log_filename].append(loss)
                epoch_dict[log_filename].append(epoch)
                accuracy_dict[log_filename].append(accuracy)
        result = regex2.search(line)
        if result:
            lr = result[1]
            learning_rate_dict[log_filename]=lr
        result = regex3.search(line)
        if result:
            validation_accuracy = result[2]
            validation_accuracy_dict[log_filename].append(float(validation_accuracy))

    
#plt.legend(handles = lines)

def plot_loss_curves():
    for log_filename in loss_dict:
        plt.plot(epoch_dict[log_filename], loss_dict[log_filename], label = "lr="+learning_rate_dict[log_filename])
    plt.legend()
    plt.title("Learning curves for different values of the learning rate")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.show()

def plot_accuracy_curves():
    f, axarr = plt.subplots(2, sharex=True)
    for log_filename in loss_dict:
        axarr[0].plot(epoch_dict[log_filename], accuracy_dict[log_filename], label = "lr="+learning_rate_dict[log_filename])
        if(len(validation_accuracy_dict[log_filename])==len(accuracy_dict[log_filename])): #Plot the validation accuracy when available
            axarr[1].plot(epoch_dict[log_filename], validation_accuracy_dict[log_filename], label = "lr="+learning_rate_dict[log_filename])
    plt.legend()
    plt.title("Learning curves for different values of the learning rate")
    axarr[0].set_ylabel("training accuracy")
    plt.xlabel("epoch")
    axarr[1].set_ylabel("validation accuracy")
    plt.show()

#Confusion Matrix
#https://codeyarns.com/2014/10/24/how-to-create-a-confusion-matrix-plot-using-matplotlib/
import numpy as np
import os.path
import seaborn as sns
def plot_confusion_matrix():
    for log_filename in loss_dict:
        confusion_matrix_filename = log_filename.replace(".txt","_confusion_matrix.txt")
        if os.path.isfile(confusion_matrix_filename):
            matrix = np.loadtxt(confusion_matrix_filename).astype(int)
            f, ax = plt.subplots(figsize=(9, 6))
            sns.heatmap(matrix, annot=True, fmt="d", linewidths=.5, ax=ax, xticklabels = ["no","yes"], yticklabels = ["no","yes"])
            plt.ylabel("Predicted label")
            plt.xlabel("Target label")
            plt.title(log_filename)
            plt.show()

plot_loss_curves()
plot_accuracy_curves()
plot_confusion_matrix()

