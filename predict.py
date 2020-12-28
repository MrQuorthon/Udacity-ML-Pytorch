# Imports

## %matplotlib inline
## %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session
import json
import helper
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import argparse
import train



# Argparse config section

parser = argparse.ArgumentParser(description='Testing a Neural Network with the test sample')
parser.add_argument('--checkpoint_path', type=str,
                    help='path to recover and reload checkpoint',default='checkpoint.pth')
parser.add_argument('--image_path', type=str,
                    help='/path/to/image',default='flowers/test/102/image_08023.jpg')
parser.add_argument('--top_k', type=int,
                    help='top k: top categories by prob predictions',default=5)
parser.add_argument('--cat_to_name', type=str,
                    help='category name mapping',default='cat_to_name.json')
parser.add_argument('--device', type=str,
                    help='Choose -cuda- gpu or internal -cpu-',default='cuda')
parser.add_argument('--network', type=str,
                    help='Torchvision pretrained model. May choose densenet121 too', default='vgg19')
parser.add_argument('data_dir', type=str,
                    help='Path to root data directory', default='/flowers/')

args = parser.parse_args()
checkpoint_path = args.checkpoint_path
image_path = args.image_path
top_k = args.top_k
device = args.device
cat_to_name = args.cat_to_name
network = args.network
data_dir = args.data_dir



# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)



# Loading the checkpoint

def Reload_model(checkpoint_path, network):
    if network == 'vgg19':
        model = models.vgg19(pretrained = True)
    else:
        model = models.densenet121(pretrained = True) 
    checkpoint = torch.load(checkpoint_path)
    classifier = checkpoint['classifier']
    optimizer = checkpoint['optimizer']
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    epochs = checkpoint['epochs']    
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    return model


# Inference for classification

## Image Preprocessing
def process_image(image_path): 
    image = Image.open(image_path)
    transform_image = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    image_transf = transform_image(image)
    
    return image_transf


## Class Prediction
def predict(image_path, checkpoint_path, network, top_k, device):
    model = Reload_model(checkpoint_path, network)
    image = process_image(image_path)
    if device == 'cuda':
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        image = torch.from_numpy(image).type(torch.FloatTensor)    
    image = image.unsqueeze_(0)

    model.to(device)
    image = image.to(device)
    model.eval()
    
    with torch.no_grad():
        logps = model.forward(image)
    
    ps = F.softmax(logps,dim=1)
    probs, classes = ps.topk(top_k)
    probs = np.array(probs[0][0])
    classes = [cat_to_name[str(index + 1)] for index in np.array(probs[1][0])]
    return probs, classes 



def main():
    Reload_model(checkpoint_path, network)
    process_image(image_path)
    predict(checkpoint_path, image_path, network, top_k, device) 
    
    
    
if __name__== "__main__":
    main()
