"""
    This tutorial will covert two topics:
    1- How to view the model summary (ie: Number of parameters for every layer)
    2- Applying a tensor hook to get input size for every layer 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T




model = torchvision.models.resnet50(pretrained=True)


#pip install torchinfo
from torchinfo import summary

summary(model, input_size=(3, 100, 100)) # batch_size, channels, H, W

