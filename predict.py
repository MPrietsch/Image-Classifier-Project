import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import pandas as pd
import argparse
from torch.autograd import Variable
import json
import sys
import os, random

# Don't think I'll need it here
from collections import OrderedDict
import seaborn as sb
import torch.nn.functional as F

#####

def dataparser():
    parser = argparse.ArgumentParser(description = 'File prediction')

    parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'Dataset Directory')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Checkpoint')
    parser.add_argument('--gpu', type = bool, default = 'True', help = 'By default, parser will use CUDA if True, and CPU if False.')
    parser.add_argument('--top_k', type = int, default = 5, help = 'Top K most likely classes')
    parser.add_argument('--image', type = str, default = 'default', help = 'Path to default random image')
    parser.add_argument('--json_file', type = str, default = 'cat_to_name.json', help = 'File for mapping flower names')

    arguments = parser.parse_args()
    return arguments

#####

args = dataparser()

with open(args.json_file, 'r') as f:
    cat_to_name = json.load(f)

default_device = args.gpu
use_gpu = torch.cuda.is_available()
device = torch.device("cpu")
if default_device and use_gpu:
    device = torch.device("cuda:0")
    print(f"Device is set to {device}")
else:
    device = torch.device("cpu")
    print(f"Device is set to {device}")

#####

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'pretrained vgg19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False


    model.class_to_idx = checkpoint['model_index']

    model.classifier = checkpoint['classifier']

    model.optimizer = checkpoint['optimizer']

    model.load_state_dict(checkpoint['state_dict'])

    return model

#####

def process_image(image):
    pil_image = Image.open(image)

    if pil_image.width > 256 or pil_image.height > 256:
        if pil_image.height < pil_image.width:
            factor = 256 / pil_image.height
        else:
            factor = 256 / pil_image.width
        pil_image = pil_image.resize((int(pil_image.width * factor), int(pil_image.height * factor)))

    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))

    np_image = np.array(pil_image) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean) / std

    np_image = np_image.transpose((2, 0, 1))

    return np_image

#####

def predict(image_path, model, device, cat_to_name, topk):

    model.to(device)

    img = process_image(image_path)

    if torch.cuda.is_available():
        image = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    else:
        image = torch.from_numpy(img).type(torch.FloatTensor)


    image = image.unsqueeze(0)

    probs = torch.exp(model.forward(image))
    prob_arr, top_classes = probs.topk(topk)

    prob_arr = prob_arr.detach().cpu().numpy().tolist()[0]
    top_classes = top_classes.detach().cpu().numpy().tolist()[0]

    idx_to_class = {v: k for k, v in
                    model.class_to_idx.items()}

    pred_labels = [idx_to_class[label] for label in top_classes]
    pred_class = [cat_to_name[idx_to_class[label]] for label in top_classes]

    return prob_arr, pred_labels, pred_class

#####

model = load_checkpoint(args.checkpoint)

random_set = args.data_dir + '/' + random.choice(os.listdir(args.data_dir)) + '/'
random_path = random_set + random.choice(os.listdir(random_set)) + '/'

if args.image == 'default':
    loaded_image = random_path + random.choice(os.listdir(random_path))
else:
    loaded_image = args.image

predict_image = process_image(loaded_image)

topk_probs, topk_labels, topk_classes = predict(predict_image, model, device, cat_to_name, args.top_k)

print('Predicted top classes : ', topk_classes)
print('Flowers: ', topk_labels)
print('Probablity: ', topk_probs)
