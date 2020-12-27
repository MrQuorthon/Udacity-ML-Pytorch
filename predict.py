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
    checkpoint = torch.load(checkpoint_path)
    if network == 'vgg19':
        model = models.vgg19(pretrained = True)
    else:
        model = models.densenet121(pretrained = True) 

    model.class_to_idx = checkpoint['class_to_idx']
    classifier.load_state_dict(checkpoint['classifier'])
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']    
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    return model



# Inference for classification

## Image Preprocessing
def process_image(image_j): 
    image_j = (image_path) 
    image_import = Image.open(image_j)
    transform_image = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    image_transf = transform_image(image_import)
    
    return image_transf


## Class Prediction
def predict(image_path, checkpoint_path, network, top_k, device):
    model = load_checkpoint(checkpoint_path, network)
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
    return probs, classes 


## Sanity Checking
def sanity_check():
    model = Reload_model(checkpoint_path, network)

    plt.rcParams['figure.figsize'] = (10,6)
    plt.subplot(211)
    
    index = 1
    path = image_path

    probs = predict(path, model)
    image = process_image(path)

    axs = imshow(image, ax = plt)
    axs.axis('off')
    axs.title(cat_to_name[str(index)])
    axs.show()
    
    j = np.array(probs[0][0])
    k = [cat_to_name[str(index + 1)] for index in np.array(probs[1][0])]
        
    N=float(len(k))
    fig,ax = plt.subplots(figsize=(10,6))
    width = 0.9
    tickLocations = np.arange(N)
    ax.bar(tickLocations, j, width, linewidth=5.0, align = 'center')
    ax.set_xticks(ticks = tickLocations)
    ax.set_xticklabels(k)
    ax.set_xlim(min(tickLocations)-0.75,max(tickLocations)+0.75)
    ax.set_yticks([0.25,0.5,0.75,1])
    ax.set_ylim((0,1))
    ax.yaxis.grid(True)

    plt.show()

sanity_check()

def main():
    Reload_model(checkpoint_path, network)
    process_image(image)
    imshow(image, ax=None, title=None)
    predict(checkpoint_path, image_path, image, network, top_k, device)
    sanity_check()


if __name__== "__main__":
    main()
