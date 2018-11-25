#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56
#Started from a tutorial on the web.
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)
# Hyper parameters
num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001


# MNIST dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())


n_train = 45000
indices = list(range(len(train_dataset)))
random.shuffle(indices)


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           sampler=SubsetRandomSampler(indices[:n_train]))
                                        
validation_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           sampler=SubsetRandomSampler(indices[n_train:]))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


# https://arxiv.org/pdf/1412.6806.pdf
#
# (Attempt at 2 Pytorch implementation of the paper : STRIVING FOR SIMPLICITY - THE ALL CONVOLUTIONAL NET
# JostTobiasSpringenberg∗,AlexeyDosovitskiy∗,ThomasBrox,MartinRiedmiller
# Department of Computer Science - University of Freiburg - Freiburg, 79110, Germany

# Based on the code from the PyTorch Tutorial.

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=0), #Adapted to CIFAR here. 3 channels instead of one
            nn.BatchNorm2d(96),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU())
           # nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4*4*192, num_classes) #Adapted to CIFAR here. 8x8 instead of 7x7 (32x32 images instead of 28x28)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)


#Logging functions. Will be used later for ploting graphs
import time
import datetime
logfile_prefix = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')

logfile = open("results/" + logfile_prefix + ".txt","w+")
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model, file=logfile)
print(optimizer, file=logfile)
print("Learning rate:", learning_rate, file=logfile)
logfile.flush()

# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    correct = 0
    total = 0
    validation_total = 0
    validation_correct = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        
        if (i+1) % (batch_size/2) == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy:{:.2f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), correct/total), file=logfile)
            logfile.flush()
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy:{:.2f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), correct/total))
    #At the end of the epoch, perform validation.
    with torch.no_grad():
        validation_correct = 0
        validation_total = 0
        for images, labels in validation_loader:
            validation_images = images.to(device)
            validation_labels = labels.to(device)
            validation_outputs = model(validation_images)
            _, validation_predicted = torch.max(validation_outputs.data, 1)
            validation_total += validation_labels.size(0)
            validation_correct += (validation_predicted == validation_labels).sum().item()
        print ("Epoch {} validation accuracy= {:.4f}".format(epoch+1, validation_correct/validation_total))
        print ("Epoch {} validation accuracy= {:.4f}".format(epoch+1, validation_correct/validation_total), file=logfile)
            
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

w, h = 10,10
confusion_matrix = [[0 for x in range(w)] for y in range(h)] 

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted = predicted.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        #Builds the confusion matrix
        for prediction, target in zip(predicted,labels):
            confusion_matrix[prediction][target]+=1


    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total), file=logfile)
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

np.savetxt("results\\"+logfile_prefix+"_confusion_matrix.txt",np.matrix(confusion_matrix))

#Just for the log
print("Predicted (row) labels vs targets (column)", file=logfile)
for i in range(0,10):
    for j in range(0,10):
        print(confusion_matrix[i][j],"\t",end='', file=logfile)
    print("\n",end="", file=logfile)

logfile.close()
