
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








