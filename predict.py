
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
# from train.py import test_dataset
from torch import nn
from torch import optim
import torch.nn.functional as y
from torchvision import datasets as d
from torchvision import transforms as t
from torchvision import transforms as t
from torchvision.transforms import Compose as c
from collections import OrderedDict
import json
from PIL import Image
import argparse
import sys

# python predict.py flowers/test/1/image_06743 checkpoint.pth
parser= argparse.ArgumentParser()
parser.add_argument("path", help="path to image ")
parser.add_argument("checkpoint", help="path to checkpoint")
parser.add_argument("--top_k",default=3, help="top k classes")
parser.add_argument("--category_name",default="cat_to_name.json", help="category names ")
parser.add_argument("--gpu" , action='store_true',help="use gpu for training")
parser.parse_args()

gpu= parser.parse_args().gpu
chkpnt= parser.parse_args().checkpoint
image_path= parser.parse_args().path

category_name=parser.parse_args().category_name
top_k=parser.parse_args().top_k
device= torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

chkpnt = chkpnt if ".pth" in chkpnt else f"{chkpnt.strip()}.pth"


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
    
def load(file):
    statedict=torch.load(file)
    model=dl(statedict ["arch"])
    model.classifier=statedict['classifier']
    model.load_state_dict (statedict ['state_dict'])
    for param in model.parameters():
            param.requires_grad = False
    return model
            

test_data=c([
        t.Resize(255),
        t.CenterCrop(224),
        t.ToTensor(),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_dataset=d.ImageFolder( 'flowers/test', transform=test_data)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model

    # Converting image to PIL image using image file path
    img = Image.open(f'{image}' + '.jpg')

    # Building image transform
    transform = c([t.Resize(256),
                   t.CenterCrop(224),
                   t.ToTensor(),
                   t.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])])
    
    ## Transforming image for use with network
    timg = transform(img)
    
    # Converting to Numpy array
    return  np.array(timg)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
   
    img = process_image(image_path)
    
    # Convert image to PyTorch tensor first
    img = torch.from_numpy(img)
    
    img = img.unsqueeze(0)
    
    img.to(device)
    output = model(img)
    
    prob= torch.exp(output)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_prob, top_indices = prob.topk(topk)
    
    # Convert to lists
    top_prob = top_prob.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
    
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    
    idx_to_class = {value: key for key, value in test_dataset.class_to_idx.items()}
    #print(idx_to_class)
    
    top_class = [idx_to_class[index] for index in top_indices]
    
    return top_prob, top_class

def display_stats( image_path, cat_to_nam):
    # Getting prediction
    probs,classes = predict(image_path, model, topk=top_k)

    # Uncomment to check if the class of flower (as seen from the file path) matches the top one predicted:
    #print(classes[0])

    # Converting classes to names
    names = []
    for i in classes:
        names += [cat_to_name[i]]

    # Creating PIL image
    image = Image.open(f'{image_path}.jpg')

    # Plotting test image and predicted probabilites
    f, ax = plt.subplots(2,figsize = (6,10))

    ax[0].imshow(image)
    ax[0].set_title(names[0])

    y_names = np.arange(len(names))
    ax[1].barh(y_names, probs)
    ax[1].set_yticks(y_names)
    ax[1].set_yticklabels(names)
    ax[1].invert_yaxis()

    plt.show()


# Uncomment below to see the transformed image on which the model based its prediction
#imshow(process_image(image_path), ax=None, title=None)
    
import json

def j():
    with open(category_name,'r') as f:
        cat_to_name = json.load(f)
        pass
    return cat_to_name


model=load(chkpnt).to(device)
cat_to_name=j()

display_stats( image_path, cat_to_name)


