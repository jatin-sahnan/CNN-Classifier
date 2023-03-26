#!/usr/bin/env python
# coding: utf-8

# # <Font Color=Dark>CNN Classifier</Font>

# <hr style="border:2px solid gray">

# ### <Font Color=Grey>3) Creating a CNN classifier to classify the pathology images as foreground vs background. The CNN architecture: CONV layer with 163x3 filters with pad 1, stride1, RELU, POOL 2x2 with stride2, CONV layer with 83x3 filters with pad1, stride1, RELU, POOL 2x2 with stride2, Dense layer of size 64,RELU.</Font>

# <hr style="border:2px solid gray">

# In[2]:


# Importing Basic Libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Importing Torch Libraries

import torch
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import datasets, transforms, models


# In[5]:


torch.device('cpu' if not torch.cuda.is_available() else 'cuda')


# In[6]:


# Setting train and test data path

train_path = 'C:/Users/jatin/Desktop/pathologyData/Train'
test_path= 'C:/Users/jatin/Desktop/pathologyData/Test'


# In[8]:


#Importing the datasets

from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms, models

data_train = ImageFolder(train_path,transform = transforms.Compose([
    transforms.Resize((100,100)),transforms.ToTensor(),transforms.RandomHorizontalFlip(),]))

data_test = ImageFolder(test_path,transforms.Compose([
    transforms.Resize((100,100)),transforms.ToTensor(),transforms.RandomHorizontalFlip(),]))


# In[10]:


# checking train data

data_train


# In[11]:


# Checking test data

data_test


# In[12]:


# Checking for classes present

data_train.classes


# In[13]:


# Loading dataset for model in batch size of 64

batch_size = 64

train_path_img = DataLoader(data_train, batch_size, shuffle = True, num_workers = 2)
test_path_img = DataLoader(data_test, batch_size, num_workers = 2)


# In[14]:


# Creating the NN

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 16 filters of 3x3 size
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1) # 8 filters of 3x3 size
        self.fc1 = nn.Linear(25*25*8, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # BLOCK 1: CONV + MAXPOOL + RELU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # BLOCK 2: CONV + MAXPOOL + RELU
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        # FLATTEN
        x = x.flatten(start_dim=1)
        # BLOCK 3: FC + RELU
        x = F.relu(self.fc1(x))
        # BLOCK 4: FC + LOG SOFTMAX
        x = F.relu(self.fc2(x))
        return x

model = CNN()
print(model)


# In[17]:


# Creating function for calculating loss & Accuracy

def train(model, data_loader, optimizer, criterion, epoch):
    model.train()
    loss_train = 0
    num_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        prediction = output.argmax(dim=1)
        num_correct += prediction.eq(target).sum().item()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAccuracy: {:.0f}%'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss_train / (batch_idx + 1),
                100. * num_correct / (len(data) * (batch_idx + 1))))
    loss_train /= len(data_loader)
    accuracy = num_correct / len(data_loader.dataset)
    return loss_train, accuracy


def test(model, data_loader, criterion):
    model.eval()
    #loss_test = 0
    loss=0
    #num_correct = 0
    num=0
    with torch.no_grad():
        for data, target in data_loader: 
            data, target = data, target
            output = model(data)
            loss_1 = criterion(output, target)
            loss += loss_1.item()  
            prediction = output.argmax(dim=1)
            num += prediction.eq(target).sum().item()
    loss /= len(data_loader)
    accuracy = num / len(data_loader.dataset)
    return loss, accuracy


# In[19]:


# Initiating the neural network
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#getting Loss and accuracy for training dataset
for epoch in range(1, 6):
    loss_train, acc_train = train(model, train_path_img, optimizer, criterion, epoch)
    print('Epoch {} Train: Loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
        epoch, loss_train, 100. * acc_train))


# In[20]:


#getting Loss and accuracy for test dataset
for epoch in range(1, 6):
    loss_test, acc_test = test(model, test_path_img, criterion)
    print('Epoch {} Test: Loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
        epoch, loss_test, 100. * acc_test))


# In[21]:


# Getting accuracies for each Epoch

index_values = ['epoch 1', 'epoch 2', 'epoch 3', 'epoch 4', 'epoch 5']
column_values = ['Test_Accuracy']

df_a1 = pd.DataFrame(data = acc_test, 
                  index = index_values, 
                  columns = column_values)
df_a1


# In[22]:


# Getting mean & std dev for test accuracy

import statistics as st

print("The mean of Test accuracy after 1st run is: %2d%%"
                              %(st.mean(df_a1.Test_Accuracy)*100))
                               
print("The Standard Deviation of Test accuracy after 1st run is: %s"
                              %(round(st.stdev(df_a1.Test_Accuracy)*100,3)))


# #### Second Run

# In[23]:


#getting Loss and accuracy for test dataset
for epoch in range(1, 6):
    loss_test, acc_test2 = test(model, test_path_img, criterion)
    print('Epoch {} Test: Loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
        epoch, loss_test, 100. * acc_test))


# In[24]:


# Getting accuracies for each Epoch

index_values = ['epoch 1', 'epoch 2', 'epoch 3', 'epoch 4', 'epoch 5']
column_values = ['Test_Accuracy']

df_a1 = pd.DataFrame(data = acc_test2, 
                  index = index_values, 
                  columns = column_values)
df_a1


# In[26]:


# Getting mean & std dev for test accuracy

print("The mean of Test accuracy after 2nd run is: %2d%%"
                              %(st.mean(df_a1.Test_Accuracy)*100))
                               
print("The Standard Deviation of Test accuracy after 2nd run is: %s"
                              %(round(st.stdev(df_a1.Test_Accuracy)*100,3)))


# #### Third Run

# In[27]:


#getting Loss and accuracy for test dataset
for epoch in range(1, 6):
    loss_test, acc_test3 = test(model, test_path_img, criterion)
    print('Epoch {} Test: Loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
        epoch, loss_test, 100. * acc_test))


# In[28]:


# Getting accuracies for each Epoch

index_values = ['epoch 1', 'epoch 2', 'epoch 3', 'epoch 4', 'epoch 5']
column_values = ['Test_Accuracy']

df_a1 = pd.DataFrame(data = acc_test3, 
                  index = index_values, 
                  columns = column_values)
df_a1


# In[29]:


# Getting mean & std dev for test accuracy

print("The mean of Test accuracy after 3rd run is: %2d%%"
                              %(st.mean(df_a1.Test_Accuracy)*100))
                               
print("The Standard Deviation of Test accuracy after 3rd run is: %s"
                              %(round(st.stdev(df_a1.Test_Accuracy)*100,3)))


# <hr style="border:2px solid gray">
