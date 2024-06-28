
**ResNet-50 Model Description**
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ResNet is a neural network architecture used for deep learning computer vision applications like object detection. It is considered to have more number of convolutional Layers and we have different categories on the basis of layers according to the depth of the network like ResNet-18,ResNet-50 and ResNet-152.

These architectures are structured by the Microsoft Researchers, in which they introduced the model with residual blocks to overcome the bottleneck condition and it contains a bunch of different terms like skip connections ,identity function ,etc.
The 34-layer plain network has a training error more than an 18-layer plain network, which was caused by degradation as we go deeper into the plain network. With the use of ResNet, when a shortcut connection was added, the error decreased for more layers and performed well in generalizing the validation data.


**Project Introduction**
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In this project , I prepare a ResNet-50 model architecture from scratch and adding layers one by one to  make it a whole Pytroch model . ResNet network helps to reduce the vanishing of gradient descent , which means when we increasing the layers in the model the growth of accuracy percentage not went on decreasing but remain stable or even better. 

In this we have total 16 residual blocks in which one block contains the 1x1 Convoultional layer, Batch_Norm layer-1, 3x3 
Convolutional layer, Batch_Norm layer-2, 1x1 Convoulational layer, Batch_Norm layer-3 and a ReLU layer.

![Screenshot 2024-06-28 145445](https://github.com/Radhika-00804/ResNet_50-model_from_scratch/assets/163717432/288f7f31-4d22-4c26-a008-783e65c71ea5)

This shows only the overview of the one residual block and in this ResNet model , we continually replicate the same block but with different size and features, and at last we get the flatten layer which gives us the coordinates of the object detected in an image in a list form [x,y,height,width].

**Importing libraries**
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
import time
from customDataset import CustomDataset
import logging
```
In this , we see the importing of logging which helps to track the events that happen when some software runs. Logging is important for software developing, debugging, and running. If you don’t have any logging record and your program crashes, there are very few chances that you detect the cause of the problem. Also,
it have this one line `from customDataset import CustomDataset` , it means we import the file named customDataset as it helps to get the data on which the model will be trained after transforming and normalizing it.

**Transform and Normalize**
```
def __init__(self, csv_file, root_dir, transform=None,normalize=True):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.normalize = normalize
        self.coord_max = self.data.iloc[:, 2:].max().max()  # Assuming maximum value for normalization
```
This file returns the annotated coordinates in the form of Tensor , as our model is made on pytorch and the model requires data is in number/Tensor form to recognize.

**Data Augmentation**
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Data augmentation is a process that artificially generates new data from existing data to train machine learning models. It's an important step when building datasets because small datasets can cause machine learning models to "overfit".
```
# Define the data augmentation transforms
    train_transforms = transforms.Compose([
                       transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
                       transforms.RandomHorizontalFlip(),  # Random horizontal flip
                       transforms.RandomRotation(10),      # Random rotation up to 10 degrees
                       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])

    test_transforms = transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    train_dataset = CustomDataset(csv_file="m_train.csv", root_dir="Path of your root directory", transform=train_transforms)
    test_dataset = CustomDataset(csv_file="m_train.csv", root_dir="Path of your root directory", transform=test_transforms)
```

**Identity Blocks**
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The identity block is a fundamental building block in the ResNet-50 architecture that allows signals to flow directly between blocks in both directions. It's a standard block in ResNets that corresponds to when the input and output activations have the same dimensions.

https://www.researchgate.net/publication/345342548/figure/fig5/AS:983188558057473@1611421827109/Building-blocks-for-the-ResNet50-A-Identity-block-and-B-Convolutional-block.png
In this , shown below , it is a residual block with identity function in it:-

```
class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride =1 , padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size =3, stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1, stride =1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
```
if, we pass nothing to class then downsample = None , as result identity will not changed.

When we pass downsample = "some convolution layer" as class constructor argument, It will downsample the identity via passed convolution layer to sucessfully perform addition. this layer will downsample the identity through code as mentioned
```
  if self.downsample is not None:
        identity = self.downsample(x)
```
