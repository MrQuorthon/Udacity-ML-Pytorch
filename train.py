#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
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



# Argparse config section

parser = argparse.ArgumentParser(description='Training a Neural Network for image selection')
parser.add_argument('data_dir', type=str,
                    help='Path to root data directory', default='/flowers/')
parser.add_argument('--network', type=str,
                    help='Torchvision pretrained model. May choose densenet121 too', default='vgg19')
parser.add_argument('--lr', type=float,
                    help='Learning rate', default=0.00025)
parser.add_argument('--hidden_units', type=int,
                    help='Input for hidden units. If densenet, must be below 1024', default=4096)
parser.add_argument('--epochs', type=int,
                    help='Number of epochs to run', default=12)
parser.add_argument('--device', type=str,
                    help='Choose -cuda- gpu or internal -cpu-', default='cpu')
parser.add_argument('--save_dir', type=str,
                    help='path to directory to save the checkpoints',default='checkpoint.pth')

args = parser.parse_args()
data_dir = args.data_dir
network = args.network
lr = args.lr
hidden_units = args.hidden_units
epochs = args.epochs
device = args.device
save_dir = args.save_dir
   

# TODO: Load and transform the datasets

def data_load_transform(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=24)

    return train_data, train_loader, valid_loader, test_loader


# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# Building and training the classifier

## Building the classifier
def modelbuilder(device, network, hidden_units, lr):
    if network == 'vgg19':
        model = models.vgg19(pretrained = True)
    else:
        model = models.densenet121(pretrained = True) 

    if torch.cuda.is_available() and device == 'cuda':
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    if network == 'vgg19':
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(1024, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 102),
                                     nn.LogSoftmax(dim=1))
    else:
        model.classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 102),
                                     nn.LogSoftmax(dim=1))

    return model


## Training the classifier
def modeltrainer(epochs, data_dir, device, network, hidden_units, lr):
    steps = 0
    running_loss = 0
    print_every = 50
    train_losses, test_losses = [], []
    model = modelbuilder(device, network, hidden_units, lr)
    train_data, train_loader, valid_loader, test_loader = data_load_transform(data_dir)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    if torch.cuda.is_available() and device == 'cuda':
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
    
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                train_losses.append(running_loss/len(train_loader))
                test_losses.append(test_loss/len(valid_loader))
                               
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/len(train_loader):.3f}.. "
                      f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
    return model


# Save the checkpoint
def checkpoint(data_dir, save_dir, device, network, hidden_units, lr):
    train_data, train_loader, valid_loader, test_loader = data_load_transform(data_dir)
    model = modelbuilder(device, network, hidden_units, lr)
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    state = {
            'epochs': epochs,
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'class_to_idx':model.class_to_idx
            }
    torch.save(state, save_dir)


# Define Main
def main():
    data_load_transform(data_dir)
    modelbuilder(device, network, hidden_units, lr)
    modeltrainer(epochs, data_dir, device, network, hidden_units, lr)
    checkpoint(data_dir, save_dir, device, network, hidden_units, lr)


if __name__== "__main__":
    main()
