from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
class GoogleStreetView(Dataset):
    def __init__(self, csv_path, transform=None, labels=None):
        self.df = pd.read_csv(csv_path)
        self.tranform = transform
        classes = self.df[:,'class'].unique()
        if(labels):
            self.labels = labels
        else:
            self.labels = {}
            for c in classes:
                self.labels[c] = int(c) - 5001
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.df.loc[idx, 'dir']
        image = io.imread(img_name)
        label = self.labels[self.df.loc[idx, 'class']]
        if(self.tranform):
            image = self.tranform(image)
        return (image, label)
