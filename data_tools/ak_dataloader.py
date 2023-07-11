import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

class ak_dataset(Dataset):
    """dataset for Animal Kingdom"""

    def __init__(self, csv_file, root_dir, transform=None):
        self.cwd = "/home/jonathan/Desktop/Perona_Research"
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])

        image = io.imread(img_name)

        return image

def get_data(batch_size = 100, num_workers = 8):
    cwd = "/home/jonathan/Desktop/Perona_Research"

    train_dataset = ak_dataset(
        csv_file = cwd + "/data_tools/PE_data_ak_P3_mammal.csv",
        root_dir = cwd + "/datasets/Animal_Kingdom/action_recognition/dataset/"
    )

    val_dataset = ak_dataset(
        csv_file = cwd + "/data_tools/PE_data_ak_P3_mammal.csv",
        root_dir = cwd + "/datasets/Animal_Kingdom/action_recognition/dataset/"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = True
    )

    val_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = True
    )

    return train_loader, val_loader

