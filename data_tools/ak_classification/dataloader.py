import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

# the structure of this dataset will follow CIFAR10 where __getitem__ returns a tuple of (image, target) where target is the index of the target class
class ak_classification_dataset(Dataset):
    """dataset for Animal Kingdom"""

    def __init__(self, csv_file, root_dir, animal_label, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.label_to_int = {}
        if animal_label == "animal_parent_class":
            self.label_to_int = {
                "Reptile": 0,
                "Bird": 1,
                "Mammal": 2,
                "Amphibian": 3,
                "Fish": 4
            }

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # this dictionary converts the string labels to integers
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        
        label = self.label_to_int[self.landmarks_frame.iloc[idx, 1]]

        image = PIL.Image.open(img_name, mode="r")
        # have to use PIL instead of io.imread because transform expects PIL image
        # image = io.imread(img_name)
        
        print(type(image))
        
        if self.transform is not None:
            image = self.transform(image)

        image = np.reshape(image, (3, 360, 640))
        image = image.to(torch.float32)

        return (image, label)

def get_data(batch_size=100, num_workers=8):
    # cwd = "/home/jonathan/Desktop/Perona_Research"
    cwd = "/Users/jonathanlin/Documents/GitHub/research_transfer/"
    # for the lab computer directory
    if torch.cuda.is_available():
        cwd = "/home/jonathan/Desktop/Perona_Research/"

    train_dataset = ak_classification_dataset(
        csv_file = cwd + "data_tools/ak_classification/dataset_train.csv",
        root_dir = cwd + "datasets/Animal_Kingdom/pose_estimation/dataset/",
        animal_label = "animal_parent_class",
        transform=torchvision.transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.RandAugment(),
                         transforms.ToTensor()
                     ])
    )

    val_dataset = ak_classification_dataset(
        csv_file = cwd + "data_tools/ak_classification/dataset_test.csv",
        root_dir = cwd + "datasets/Animal_Kingdom/pose_estimation/dataset/",
        animal_label = "animal_parent_class",
        transform=torchvision.transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.RandAugment(),
                         transforms.ToTensor()
                     ])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = False
    )

    return train_loader, val_loader