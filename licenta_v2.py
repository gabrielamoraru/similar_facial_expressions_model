

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:14:18 2024

@author: Gabi
"""
import time
import torch
import dlib
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir,  transform=None, target_transform=None ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
   
   
    def __getitem__(self, idx):
        img_path1 = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path2 = os.path.join(self.img_dir, self.img_labels.iloc[idx, 5])
        image1 =  Image.open(img_path1).convert('RGB')
        image2 =  Image.open(img_path2).convert('RGB')
        label = self.img_labels.iloc[idx, 10]
       
        if self.transform:
            image1 = self.transform(image1)
            image2= self.transform(image2)
        if self.target_transform:
            label = self.target_transform(label)
        image = torch.cat((image1, image2), dim=0)
        return image, label
        
        
device = torch.device('cuda')
print(device)

# Define a data transformation (resize, normalize, etc.) if needed
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

class Net(nn.Module):
    def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(6, 6, 3)
       self.pool = nn.MaxPool2d(2, 2)
       self.conv2 = nn.Conv2d(6, 16, 3)
       self.fc1 = nn.Linear(16 * 54 * 54, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


training_data=CustomImageDataset('./train_dataset/train_dataset.csv', 'train_dataset',transform=transform)
test_data=CustomImageDataset('./test_dataset/dataset_test.csv', 'test_dataset',transform=transform)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

net = Net().to(device)
print(net)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

epochs = 5
min_valid_loss = np.inf


for e in range(epochs):
    train_loss = 0.0
    train_acc=0.0
    for data, labels in train_dataloader:
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
           
         
        # Clear the gradients
        optimizer.zero_grad()
        # Forward Pass
        
        target = net(data)
        
        # Find the Loss
        labels = labels.unsqueeze(1)
        labels = labels.float()
        loss = criterion(target,labels)
        # Calculate gradients 
        loss.backward()
        # Update Weights
        optimizer.step()
        train_acc = accuracy_fn(labels, torch.round(torch.sigmoid(target)))
        # Calculate Loss
        train_loss += loss.item()
    test_acc=0.0 
    valid_loss = 0.0
    net.eval()     # Optional when not using Model Specific layer
    for data, labels in test_dataloader:
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
         
        # Forward Pass
        labels = labels.unsqueeze(1)
        labels = labels.float()
        target = net(data)
        # Find the Loss
        valid_acc = accuracy_fn(labels, torch.round(torch.sigmoid(target)))
        loss = criterion(target,labels)
        # Calculate Loss
        valid_loss += loss.item()
        
    print(f'Epoch {e+1} \t\t Training Loss: { (train_loss / len(train_dataloader)):.2f} \t\t Validation Loss: {(valid_loss / len(test_dataloader)):.2f} Training_Accuracy:{train_acc: .2f} Validation_Accuracy:{valid_acc:.2f}')
     
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss}--->{(valid_loss/len(test_dataloader)):.2f}) \t Saving The Model')
        min_valid_loss = valid_loss
         
        # Saving State Dict
        torch.save(net.state_dict(), 'saved_model.pth')