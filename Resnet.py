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


logging.basicConfig(filename="prediction.log", level=logging.INFO, filemode='w',
                    format='%(asctime)s -%(levelname)s -%(message)s')                                            

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    logging.info("Starting the training process.")
    transform = transforms.Compose([transforms.ToTensor()])

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

    
    train_dataset = CustomDataset(csv_file="m_train.csv", root_dir="C:\\Users\\Radhika\\Downloads\\m_train_original", transform=train_transforms)
    test_dataset = CustomDataset(csv_file="m_train.csv", root_dir="C:\\Users\\Radhika\\Downloads\\m_train_original", transform=test_transforms)
    
    batch_size = 4
    # Ensure dataset split sizes match the total length of the dataset
    total_length = len(train_dataset)
    train_length = int(0.8 * total_length)
    test_length = total_length - train_length


    train_set, test_set = torch.utils.data.random_split(train_dataset, [train_length, test_length])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # print(f"Total dataset length: {total_length}")
    # print(f"Training set length: {train_length}")
    # print(f"Testing set length: {test_length}")

    # print(f"Training set length (verify): {len(train_set)}")
    # print(f"Testing set length (verify): {len(test_set)}")

    # Get a batch of training data
    dataiter = iter(train_loader)
    images, coordinates = next(dataiter)
    
###################################################################################################################

    # # Check the types and shapes
    # print(f"Image tensor:\n{images[0]}")
    # print(f"Image shape: {images[0].shape}")
    # print(f"Image datatype: {images[0].dtype}")
    # print(f"Coordinates: {coordinates[0]}")
    # print(f"Coordinates datatype: {coordinates[0].dtype}")

    # # Convert the first image to a NumPy array and plot it
    # img_np = images[0].numpy()
    # img_np = np.transpose(img_np, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

    # # Plot the image using matplotlib
    # plt.imshow(img_np)
    # plt.title(f'Coordinates: {coordinates[0].numpy()}')
    # plt.show()

#########################################################################################################################

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

    def forward(self,x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module): #[3,4,6,3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride =2, padding=1)

        #ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride = 1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels *4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,
                                                          stride = stride),
                                                nn.BatchNorm2d(out_channels*4))
            
            layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
            self.in_channels = out_channels*4

            for i in range(num_residual_blocks -1):
                layers.append(block(self.in_channels, out_channels))

            return nn.Sequential(*layers)

def ResNet50(img_channels=3, num_classes = 4):
    return ResNet(block, [3,4,6,3], img_channels, num_classes)

model = ResNet50().to(device)
logging.info(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()
 
    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (inputs, labels) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

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

    logging.info(f"Epoch {epoch+1} completed. Average batch inference time: {batch_inference_time*1000:.2f} ms. Loss: {running_loss/len(train_loader):.4f}")


MODEL_PATH = Path("Model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ResNet_50.pt"
MODEL_PATH_SAVE = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_PATH_SAVE}")
torch.save(model, MODEL_PATH_SAVE)


# # Manually print an image and compare coordinates
# def print_image_and_compare(dataset, model, idx):
#     image, true_coords = dataset[idx]
#     true_coords = true_coords.numpy() * dataset.coord_max  # Denormalize coordinates
#     model.eval()
#     with torch.no_grad():
#         image_tensor = image.unsqueeze(0).to(device)
#         predicted_coords = model(image_tensor).cpu().numpy().flatten()

#     image = image.permute(1, 2, 0).numpy()
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
#     image = np.clip(image, 0, 1)

#     plt.imshow(image)
#     plt.title(f'True: {true_coords}\nPredicted: {predicted_coords}')
#     plt.show()

#     error = np.abs(true_coords - predicted_coords)
#     accuracy = 100 - np.mean(error / true_coords * 100)
#     print(f'Error: {error}')
#     print(f'Accuracy: {accuracy:.2f}%')

# # Test the function
# print_image_and_compare(test_dataset, model, 0)

