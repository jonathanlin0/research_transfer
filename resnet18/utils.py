import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

plt.style.use("ggplot")

def get_data(batch_size=64):
    """
    prepares the training and validation sets and the data loaders
    """
    # CIFAR10 training dataset
    dataset_train = datasets.CIFAR10(
        root="cifar_10_data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    #CIFAR10 validation dataset
    dataset_valid = datasets.CIFAR10(
        root="cifar_10_data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # create dataloaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, valid_loader

# train_acc, valid_acc, train_loss, and valid_loss are lists containing the respective values for each epoch
def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    saving the training and loss plots
    """
    # accuracy plots
    plt.figure(figsize=(10,7))
    plt.plot(
        train_acc, color="tab:blue", linestyle="-",
        label="train accuracy"
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_loss.png'))
