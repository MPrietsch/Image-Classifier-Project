import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from torch.autograd import Variable
import argparse
import os
import sys

#####

def dataparser():
    parser = argparse.ArgumentParser(description = 'Trainer')

    parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'Dataset Directory')
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Save Checkpoint')
    parser.add_argument('--arch', type = str, default = 'vgg19', help = 'Pretrained model architecture')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Default learning rate set to 0.001')
    parser.add_argument('--epochs', type = int, default = 9, help = 'Number of epochs for the training loop')
    parser.add_argument('--gpu', type = bool, default = 'True', help = 'By default, parser will use CUDA if True, and CPU if False.')

    arguments = parser.parse_args()
    return arguments

#####

def basemodel(arch):
    if arch == None or arch == 'vgg19':
        pt_model = models.vgg19(pretrained = True)
        print('Model VGG19')
    elif arch == 'vgg19_bn':
        pt_model = models.vgg19_bn(pretrained = True)
        print('Model VGG19 with batch normalization')
    else:
        pt_model = models.vgg19(pretrained = True)
        print('Default model: VGG19')

    return pt_model

#####

args = dataparser()

default_device = args.gpu
use_gpu = torch.cuda.is_available()
device = torch.device("cpu")
if default_device and use_gpu:
    device = torch.device("cuda:0")
    print(f"Device is set to {device}")
else:
    device = torch.device("cpu")
    print(f"Device is set to {device}")

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#####

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

#####

model = basemodel(args.arch)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(4096, 4096)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(4096, 1000)),
                          ('relu3', nn.ReLU()),
                          ('dropout3', nn.Dropout(0.5)),
                          ('fc4', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


model.classifier = classifier

for param in model.parameters():
    param.requires_grad = True

print('Pretrained', args.arch,  'model modified with custom classifier' + '\n')

#####

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)

#####

def validation(model, validloader, criterion):

    test_loss = 0
    accuracy = 0
    for images, labels in validloader:

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

#####

print('Starting model training' + '\n')

epochs = args.epochs
print_every = 20
steps = 0

model.to(device)

for e in range(epochs):
    running_loss = 0
    for aa, (inputs, labels) in enumerate(trainloader):
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()

            with torch.no_grad():
                test_loss, accuracy = validation(model, validloader, criterion)

            print('Epoch: {}/{}...'.format(e+1, epochs),
                  'Training Loss: {:.3f}'.format(running_loss/print_every),
                  'Valid Set Loss: {:.3f}.. '.format(test_loss/len(validloader)),
                  'Valid Set Accuracy: {:.3f}'.format(accuracy/len(validloader)))

            running_loss = 0

            model.train()

#####

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

model.train()

print('Accuracy of the network on test set: %d %%' % (100 * correct / total))

#####

model.class_to_idx = train_dataset.class_to_idx


checkpoint = {'arch': 'pretrained vgg19',
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'model_index': model.class_to_idx}

torch.save(checkpoint, args.save_dir)
print('Trained model saved as: ', args.save_dir)

#####

print('Done')
