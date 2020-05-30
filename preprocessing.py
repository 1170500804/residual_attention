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
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)