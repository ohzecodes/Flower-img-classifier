import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn
from torch import optim
import torch.nn.functional as y
from torchvision import datasets as d
from torchvision import transforms as t
from torchvision import transforms as t
from torchvision.transforms import Compose as c
from torch.utils.data import DataLoader as DL
from collections import OrderedDict
import argparse
import sys

parser= argparse.ArgumentParser()
parser.add_argument("dir", default='flowers' ,help="Image Folder")
parser.add_argument("--arch",  default='vgg11', help="CNN Model Architecture")
parser.add_argument("--learning_rate", default=0.001, help="learning rate of the optimizer")
parser.add_argument("--hidden_units", default=2048, help="hidden units for classifier's nueral network")
parser.add_argument("--save_dir", default=".", help="checkpoint save ")
parser.add_argument("--epochs" , default=1,help="epoch of training")
parser.add_argument("--gpu" , action='store_true',help="use gpu for training")
parser.parse_args()


save_dir=parser.parse_args().save_dir
data_dir = parser.parse_args().dir
arch = parser.parse_args().arch
epochs = parser.parse_args().epochs
learning_rate=parser.parse_args().learning_rate
gpu= parser.parse_args().gpu
hidden_units=parser.parse_args().hidden_units


device= torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

print (f"using the {device}")

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



training_data =c([
    t.RandomRotation(60),
    t.RandomResizedCrop(224),
    t.RandomHorizontalFlip(),
    t.ToTensor(),
    t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_data=c([
        t.Resize(255),
        t.CenterCrop(224),
        t.ToTensor(),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_data=c([
        t.Resize(255),
        t.CenterCrop(224),
        t.ToTensor(),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

training_dataset=d.ImageFolder(train_dir, transform=training_data)
valid_dataset=d.ImageFolder(valid_dir, transform=valid_data)
test_dataset=d.ImageFolder(test_dir, transform=test_data)

train_loader=DL(training_dataset, batch_size=32, shuffle=True)
validation_loader=DL(valid_dataset,batch_size=32)
test_loader=DL(test_dataset,batch_size=32)


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# TODO: Build and train your network
# Load a pre-trained network
# (If you need a starting point, the VGG networks work great and are straightforward to use)

from torchvision import models

def dl(arch):
    if 'resnet18' in arch:
        model = models.resnet18(pretrained=True)
    elif  'alexnet' in arch:
        model = models.alexnet(pretrained=True)
    elif  "squeezenet" in arch:
        model = models.squeezenet1_0(pretrained=True)
    elif  'vgg16' in arch:
        model = models.vgg16(pretrained=True)
    elif  'densenet161' in arch:
        model= models.densenet161(pretrained=True)
    elif  'inception' in arch:
        model = models.inception_v3(pretrained=True)
    elif  'googlenet' in arch:
        model = models.googlenet(pretrained=True)
    elif  'shufflenet' in arch:
        model = models.shufflenet_v2_x1_0(pretrained=True)
    elif  "mobilenet" in arch:
        model= models.mobilenet_v2(pretrained=True)
    elif   "resnext" in arch:
        model = models.resnext50_32x4d(pretrained=True)
    elif   "wide_resnet" in arch:
        model = models.wide_resnet50_2(pretrained=True)
    elif  "mnasnet" in arch:
        model = models.mnasnet1_0(pretrained=True)
    elif "vgg11" in arch:
        model = models.vgg11(pretrained=True)
    return model
    

def buildModel(arch=arch, hidden_units=hidden_units):
    model=dl(arch)
    for param in model.parameters():
        param.requires_grad = False
    
    
    model.classifier=nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(25088,5024)),
        ('relu1',nn.ReLU()),
        ('dropout1',nn.Dropout(p=0.20)),
        ('fc2',nn.Linear(5024,hidden_units)),
        ('relu2',nn.ReLU()),
        ('dropout2',nn.Dropout(p=0.2)),
        ('fc3',nn.Linear(hidden_units,102)),
        ('output',nn.LogSoftmax(dim=1))]))
    
    return model


model = buildModel().to(device)
print(model)

criterion=nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(),lr=0.001)

def trainit(epoch=int(epochs)):
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    
    step=0
    running_loss=0
    print_every=15

    for e in range(epoch):
        for images,labels in train_loader:
            step+=1
    #         print(step)
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()

            logps=model(images)

            loss=criterion(logps,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()


            if step % print_every == 0:
                model.eval()
                test_loss=0
                accuracy=0
                for images,labels in validation_loader:
                        images,labels= images.to(device),labels.to(device)
                        logps=model(images)
                        loss=criterion(logps,labels)


                        #acurracy
                        ps= torch.exp(logps)
                        topps,top_class=ps.topk(1,dim=1)
                        equallity=top_class== labels.view(*top_class.shape)
                        accuracy+=torch.mean(equallity.type(torch.cuda.FloatTensor))

                print(f"epoch:{e+1}/{epoch}..\t"
                  f"training loss:{running_loss/print_every}..\t"
                  f"test loss:{test_loss/len(test_loader)}..\t"
                 f"test accuracy:{accuracy/len(test_loader)}")
       
trainit()

# TODO: Save the checkpoint
ckpnt={
    'state_dict': model.state_dict(),
    'classifier': model.classifier,
    'class_to_idx':test_dataset.class_to_idx,
    "arch": "vgg11"
}



torch.save(ckpnt,f"{save_dir}/checkpoint.pth")






