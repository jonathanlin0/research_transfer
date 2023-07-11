import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

def get_data(batch_size=1000, num_workers=12):
    """
    prepares the training and validation sets and the data loaders
    """
    # CIFAR10 training dataset
    dataset_train = datasets.CIFAR10(
        root=os.getcwd() + "/datasets/cifar_10_data",
        train=True,
        download=True,
        # transform=ToTensor(),
        transform =torchvision.transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=15),
            transforms.RandAugment(),
            transforms.ToTensor()
        ])
    )

    #CIFAR10 validation dataset
    dataset_valid = datasets.CIFAR10(
        root=os.getcwd() + "/datasets/cifar_10_data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # create dataloaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return train_loader, valid_loader