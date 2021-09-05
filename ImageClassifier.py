#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 15:48:30 2021

@author: neilsuji
"""

# Imports here
import torch
from torch import nn, optim, autograd
from torch import functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from collections import OrderedDict, deque
from helper import imshow
from torch.autograd import Variable

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#image transformations
data_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(255) ,
                                                  torchvision.transforms.CenterCrop(224),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                                                                    [0.229, 0.224, 0.225])])


image_datasets = torchvision.datasets.ImageFolder(data_dir, transform = data_transforms)


dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size = 64, shuffle=True)


#performing data transforms for each dataset, normalization, resize, centercrop and creating tensors for torch models
train_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(255),
                                                   torchvision.transforms.CenterCrop(224),
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                                                                    [0.229, 0.224, 0.225])])

test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(255),
                                                  torchvision.transforms.CenterCrop(224),
                                                  torchvision.transforms.ToTensor()])

valid_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(255),
                                                  torchvision.transforms.CenterCrop(224),
                                                  torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform = train_transforms)
test_dataset = torchvision.datasets.ImageFolder(test_dir, transform = test_transforms)
valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64, shuffle=True)


#use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

#creating indexing for training dataset and unpacking images/labels from trainloader
images, labels = next(iter(trainloader))
print(train_dataset.class_to_idx)
[print(labels) for labels in next(iter(trainloader))]

#open image labels, creating dictionary from JSON format
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
labels = dict(cat_to_name)

train_dataset.class_to_idx
test_dataset.class_to_idx
valid_dataset.class_to_idx

print(labels)

#train network
import torchvision.models as models


#selected vgg16 model
model = models.vgg16(pretrained=True)


model
model.to(device)

#for param in model.parameters():
    #param.requires_grad = False
#create classifier and attach it to model, replacing the previous classifier w/output size
#to match our outputs and dropout layers, relu + logsoftmax output
classifier = nn.Sequential(nn.Linear(25088, 512),
                           nn.ReLU(inplace=True),               
                           nn.Dropout(p=0.2),
                           nn.Linear(512, 224),
                           nn.ReLU(inplace=True),               
                           nn.Dropout(p=0.2),
                           nn.Linear(224, 102),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier
classifier.eval()

#define the training function for later use
def train(model, epochs, print_every):
    #choose number of epochs at function call
    epochs = epochs
    #print every x amount of images
    print_every = print_every
    #initiate total steps at 0 before starting loop
    steps = 0
    #initiate running loss at 0 before starting loop
    running_loss = 0
    criterion = nn.NLLLoss()
    #create optimizer, set params to optimize with models classifier, lr and weight decay
    optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay= 1e-2)
    for epoch in range(epochs):
        #iterate over training set and calculate loss
        for images, labels in trainloader:
            #set model to train so layers are used appropriately
            model.train()
            #collect each step as one full loop
            steps += 1
            #send images and labels to gpu if available
            images, labels = images.to(device), labels.to(device)
            #send model to gpu if available
            model.to(device)
            #initialize optimizer with 0 gradients
            optimizer.zero_grad()
            #run model on images, collect output
            logps = model(images)
            #use loss function with input and target
            loss = criterion(logps, labels)
            #perform backpropogation
            loss.backward()
            #update models params with optimizer.step
            optimizer.step()
            
            #collect running loss and store with .item for later use 
            #(perhaps mean loss over dataset/epochs)
            running_loss += loss.item() 
        
            #if training set is done looping over images set the model to eval mode 
            if steps % print_every == 0:
                #initialize losses
                valid_losses = 0
                #init accuracy
                accuracy = 0
    
                #turn off gradients because our model is not training and will
                with torch.no_grad():
                    for images, labels in validloader:
                        model.eval()
                        images, labels = images.to(device), labels.to(device)
                        #collect model output
                        logps = model(images)
                        #collect the loss input and target
                        valid_losses = criterion(logps, labels)
                        ps = torch.exp(logps)
                        #store in loss.item
                        valid_losses += valid_losses.item()
                        #unpack model output with topk probs, classes
                        top_p, top_class = ps.topk(1, dim=1)
                        #make sure labels and top class are in the same shape
                        equals = top_class == labels.view(*top_class.shape)
                        #use mean accuracy
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                    #print test accuracy and loss
                    print(f"Validation Accuracy: {accuracy/len(validloader):.3f}.."
                          f"Validation Loss : {valid_losses/len(validloader):.3f}.."
                          f"Train Loss : {running_loss/len(trainloader):.3f}")
                    running_loss = 0
                    model.train()
                    
#call to train model with the training function
train(model=model, epochs=10, print_every= 50)

def test(data):
    epochs = 15
    #initialize losses to 0
    test_losses = 0
    #create loss function
    criterion = nn.NLLLoss()
    #init accuracy to 0
    accuracy = 0
    
    #turn off gradients because our model is not training and will
    #not need to update grads
    with torch.no_grad():
        for epochs in range(epochs):
                for images, labels in data:
                    model.eval()
                    #send images and labels to device
                    images, labels = images.to(device), labels.to(device)
                    #send model to cpu because we aren't training grads
                    classifier.to(device)
                    model.to(device)
                    #collect model output
                    logps = model(images)
                    #exponentiate because model gives logarithmic output
                    ps = torch.exp(logps)
                    #collect the loss input and target
                    test_losses = criterion(logps, labels)
                    #store in loss.item
                    test_losses += test_losses.item()
                    #unpack model output with topk probs, classes
                    top_p, top_class = ps.topk(1, dim=1)
                    #make sure labels and top class are in the same shape
                    equals = top_class == labels.view(*top_class.shape)
                    #use mean accuracy
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                #print validation accuracy and loss
                print("Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                print("Test Loss: {:.3f}".format(test_losses/len(testloader)))


#function call for validation loop
test(testloader)

# Save the checkpoint
#re-create optimizer/loss function and hyperparameters used to train the model
optimizer = optim.Adam(model.classifier.parameters(), lr = 1e-3, weight_decay= 1e-2)
criterion = nn.NLLLoss()
def save_checkpoint(model, path):
    epochs = 10
    
    #save the model with torch
    checkpoint= torch.save({
                'epochs': epochs,
                'arch': 'vgg16',
                'model': model.state_dict(),
                'classifier': classifier,
                'optimizer': optimizer.state_dict(),
                'criterion': criterion,
                'mapping': test_dataset.class_to_idx
                            }, path)
    return checkpoint

save_checkpoint(model=model, path='model.pth')


#Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    
    #load the path of the model along with all important features
    checkpoint = torch.load(path)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']      
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
    class_to_idx = checkpoint['mapping']
    
    return model, checkpoint


load_checkpoint(path = 'model.pth')


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
   #resize images to shortest side of 256 with aspect ratio
    from PIL import Image
    im = Image.open(image)
    
    #resize image to 256x256
    im = im.resize((256,256))
    
    #perform center cropping with images
    width, height = im.size          
    left = (width-224)/2
    top = (height-224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    
    im = im.crop((left, top, right, bottom))
    
    #create a numpy array with the image by dividing pixels and performing normalization
    np_image = np.asarray(im, dtype=np.float32)/255
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])
    
    #transpose color channels because torch accepts different
    #color channel input
    np_image = np_image.transpose((2,0,1))
    
    
    return np_image

#make sure function call is working well with random image
image_path_ = 'flowers/test/100/image_07902.jpg'
test_image = process_image(image_path_)


test_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.transpose((1,2,0))
    
    #undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    #clip image between 0-1 or else it displays with noise
    image = np.clip(image, 0,1)
    
    ax.imshow(image)
    if title: 
        ax.set_title(title)
    
    return ax


imshow(test_image)

model.eval()

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    for param in model.parameters():
        param.requires_grad = False
    #use cuda if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #set model to eval mode to prevent it from training
    model.eval()
    model.requires_grad = False
    #obtain model output, turn it into a torch tensor
    img = torch.from_numpy(process_image(image_path)).unsqueeze(0)
    model = model.to(device)
    img = img.to(device)
    #exponentiate model output and get topk classes results
    output = torch.exp(model(img)).data.cpu()
    #unpack output with probability and classes
    probs, classes = torch.topk(output, topk)
    probs = probs.data.numpy()[0]
    
    
    # Convert classes to indices
    idx_to_class = {v: k for k,v in test_dataset.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in classes.numpy()[0]]
    
    return probs, top_classes

predict(image_path_, model)


#Display an image along with the top 5 classes
def sanity_checking(probs, classes, image):
    #index the classes again 
    probs, classes = predict(image_path_, model)
    #match image labels with top classes
    class_name = [cat_to_name[i] for i in classes]
    y = np.arange(len(class_name))
    
    #plot image and provide the top 5 probabilities for classes which it could be
    image = imshow(process_image(image_path_), ax=None, title= 'sanity check')
    fig, ax = plt.subplots(figsize = [5,5])
    plt.barh(y, probs)
    ax.set_yticks(y)
    ax.set_yticklabels(class_name)
    plt.ylabel('Flower Class')
    plt.xlabel('Probability')
    
    plt.show()
    
    
    
probs, classes = predict(image_path=image_path_, model=model)
sanity_checking(probs, classes, image=image_path_)

















