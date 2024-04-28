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
    def __init__(self, annotations_file, img_dir,  transform=None, target_transform=None, device='cuda' ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    def dlib_face_dect(self,image):
        
    #step1: read the image
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    #step2: converts to gray image
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    #step3: get HOG face detector and faces
        hogFaceDetector = dlib.get_frontal_face_detector()
        faces = hogFaceDetector(gray, 1)
        max_shape=400
        im_max=open_cv_image
    #step4: loop through each face and draw a rect around it
        for (i, rect) in enumerate(faces):
            im1=open_cv_image
            x = abs(rect.left())
            y = abs(rect.top())
            w = rect.right() - x
            h = rect.bottom() - y
            #print(x,y,w,h,i)
            #draw a rectangle
            
            if  h*w>max_shape :
                max_shape=h*w
                im_max=im1[y:h+y,x:w+x]
                #plt.figure()
                #plt.imshow(im_max)
                #print(max_shape)
           
        imageR=  cv2.cvtColor(im_max, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(imageR)
        print("IMAGE")
        #plt.figure()
        #plt.imshow(pil_image)
        return pil_image
    def __getitem__(self, idx):
        img_path1 = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path2 = os.path.join(self.img_dir, self.img_labels.iloc[idx, 5])
        image1 =  Image.open(img_path1).convert('RGB')
        image2 =  Image.open(img_path2).convert('RGB')
        label = self.img_labels.iloc[idx, 10]
        
        image3=self.dlib_face_dect(image1)
        image4=self.dlib_face_dect(image2)
        if self.transform:
            image3 = self.transform(image3)
            image4 = self.transform(image4)
        if self.target_transform:
            label = self.target_transform(label).cuda
        image = torch.cat((image3, image4), dim=0)
        return image, label
print(dlib.DLIB_USE_CUDA)
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

for epoch in range(2):  # loop over the dataset multiple times
    time=time.time()
    print(epoch)
    acc=0.0
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data)
        labels=labels.view(-1,1).float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc = accuracy_fn(labels, torch.round(torch.sigmoid(outputs)))
        # print statistics
        running_loss += loss.item()
        #print( 'input_test ', i )
        if i % 10 == 9:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}, Train acc: {acc:.2f}%')
        running_vloss = 0.0
            
            # In evaluation mode some model specific operations can be omitted eg. dropout layer
        net.train(False) # Switching to evaluation mode, eg. turning off regularisation
        for j, vdata in enumerate(test_dataloader, 0):
                vinputs, vlabels = vdata
                vinputs, vlabels= vinputs.to(device), vlabels.to(device)
                voutputs = net(vinputs)
                vlabels=vlabels.view(-1,1).float()
                vloss = criterion(voutputs, vlabels)
                vacc = accuracy_fn(vlabels, torch.round(torch.sigmoid(voutputs)))
                running_vloss += vloss.item()
               # print ('output_test',j)
        net.train(True) # Switching back to training mode, eg. turning on regularisation

        avg_loss = running_loss / 1000
        avg_vloss = running_vloss / len(test_dataloader)
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_vloss / 10:.3f}, Test acc: {vacc:.2f}%')
        running_loss = 0.0
    print("Epoch:{epoch} time:{time/360} hours")
print('Finished Training')
