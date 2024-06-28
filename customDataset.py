import torch
from torch.utils.data import Dataset
import pandas as pd 
from PIL import Image
import os
import logging

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None,normalize=True):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.normalize = normalize
        self.coord_max = self.data.iloc[:, 2:].max().max()  # Assuming maximum value for normalization
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 1])  # Assuming image path is in the second column
        image = Image.open(img_name)
        coordinates = self.data.iloc[idx, 2:].values.astype('float32')  # Assuming coordinates start from the third column

        if self.normalize:
            coordinates = coordinates / self.coord_max

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(coordinates)
