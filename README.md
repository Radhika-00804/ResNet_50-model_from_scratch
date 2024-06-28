
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

**Optimizer and Criterion**
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
An optimizer is a mathematical function or algorithm that adjusts a neural network's attributes, such as weights and learning rates, to reduce loss and improve accuracy during training. Optimizers are dependent on the model's learnable parameters, such as weights and biases.
` optimizer = torch.optim.Adam(model.parameters(),lr=0.001)`
Criterions are helpful to train a neural network. Given an input and a target, they compute a gradient according to a given loss function.
`criterion = nn.MSELoss()`

**Inference Time**
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
In a neural network, Inference time is the amount of time it takes for a model to use its learned knowledge to make predictions or evaluations based on new data. It's a critical factor in optimizing the efficiency of deep learning applications and is important for deploying models in real-world applications.
Inference time is usually measured in milliseconds and can be calculated by estimating the time required to verify all candidate solutions in different scenarios. The accuracy of the neural network often correlates with its inference time, meaning that the more time it has to make a decision, the more accurate it can be.
```
     # Measure inference time for this batch
            batch_start_time = time.time()
            outputs = model(inputs)
            batch_end_time = time.time()

                # Calculate batch inference time and average per image
            batch_inference_time = (batch_end_time - batch_start_time) / len(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=running_loss / (i + 1), inference_time=batch_inference_time * 1000)

    epoch_end_time = time.time()
    epoch_inference_time = (epoch_end_time - epoch_start_time) / len(train_loader.dataset) * 1000
```

**Save the Model**
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
After train , validate and test the model , we save the model weights and biases for further usage of the model , due to which we do not want to train the whole model again and again , if we doing different tasks related to computer vision like object detection.
```
MODEL_PATH = Path("Model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ResNet_50.pt"
MODEL_PATH_SAVE = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_PATH_SAVE}")
torch.save(model, MODEL_PATH_SAVE)
```

**Accuracy of the model**
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The accuracy of a neural network model is a metric that measures how well the model predicts positive and negative classes across all examples. It's calculated by dividing the number of correct predictions by the total number of predictions:

```
# Manually print an image and compare coordinates
def print_image_and_compare(dataset, model, idx):
    image, true_coords = dataset[idx]
    true_coords = true_coords.numpy() * dataset.coord_max  # Denormalize coordinates
    model.eval()
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        predicted_coords = model(image_tensor).cpu().numpy().flatten()

    image = image.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.title(f'True: {true_coords}\nPredicted: {predicted_coords}')
    plt.show()

    error = np.abs(true_coords - predicted_coords)
    accuracy = 100 - np.mean(error / true_coords * 100)
    print(f'Error: {error}')
    print(f'Accuracy: {accuracy:.2f}%')

# Test the function
print_image_and_compare(test_dataset, model, 0)
```

In this model , I get the 80% accuracy which is not good overall but not bad at least... it gives the coordinated with not so much difference between true_coordinates and predicted_coordinates.
