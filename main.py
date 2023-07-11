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
import random
from time import sleep
import argparse
import wandb
import os

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from torchmetrics import Accuracy
import torch.utils.data as data
from torchvision.datasets import MNIST
from torch.nn import functional as F

# import the dif models and parts
from models import beijing
from models.beijing import Block_Beijing
from models import berlin
from models.berlin import Block_Berlin

# import utility functions
from data_tools import get_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--learning_rate', default='0.001',
        help='set the learning rate of the optimizer',
        choices=['0.1', '0.01', '0.001']
    )
    parser.add_argument(
        '-o', '--optimizer', default='Adam',
        help='set the optimizer',
        choices=['Adam', 'SGD']
    )
    parser.add_argument(
        '-b', '--blocks', default='3',
        help='set the number of blocks',
        choices=['1', '2', '3', '4', '5'] # limited to 5 due to memory on this computer
    )
    parser.add_argument(
        '-f', '--activation', default='leaky_relu',
        help='set the activation function',
        choices=['relu', 'leaky_relu', 'tanh']
    )
    parser.add_argument(
        '-w', '--wandb', default='off',
        help='turn on or off wandb tracking',
        choices=['true', 'false', 't', 'f', 'on', 'off']
    )
    parser.add_argument(
        '-d', '--dropout', default='0.2',
        help='set the dropout rate',
        choices=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
    )
    parser.add_argument(
        '-bs', '--batch_size', default='1024',
        help='set the batch size',
        choices=['64','128', '256', '512', '1024', '2048']
    )
    args = vars(parser.parse_args())

    learning_rate = float(args["learning_rate"])
    num_blocks = int(args["blocks"])
    print(f"[INFO]: Set number of blocks to {num_blocks}")
    dropout = float(args["dropout"])
    print(f"[INFO]: Set the dropout to {dropout}")
    batch_size = int(args["batch_size"])
    print(f"[INFO]: Set the batch size to {batch_size}")
    if args["activation"] == "relu":
        print("[INFO]: Set activation to ReLU")
        activation = nn.ReLU(inplace=True)
    if args["activation"] == "leaky_relu":
        print("[INFO]: Set activation to LeakyReLU")
        activation = nn.LeakyReLU(inplace=True)
    if args["activation"] == "tanh":
        print("[INFO]: Set activation to Tanh")
        activation = nn.Tanh()
    if args["wandb"] == "true" or args["wandb"] == "t" or args["wandb"] == "on":
        print("[INFO]: Set wandb tracking to true")
        track_wandb = True
    if args["wandb"] == "false" or args["wandb"] == "f" or args["wandb"] == "off":
        print("[INFO]: Set wandb tracking to off")
        track_wandb = False

    # create model object
    # model = beijing.beijing(img_channels=3,
    #                   block=Block_Beijing,
    #                   num_blocks = num_blocks,
    #                   activation = activation,
    #                   lr = learning_rate,
    #                   num_classes=10)
    model = berlin.berlin(img_channels=3,
                    block=Block_Berlin,
                    num_blocks = num_blocks,
                    activation = activation,
                    lr = learning_rate,
                    dropout=dropout,
                    num_classes=10,
                    track_wandb=track_wandb)

    epochs = 100
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=learning_rate)

    if track_wandb:
        wandb.init(
            project="Perona_Research",

            config={
                "learning_rate":learning_rate,
                "architecture":"Berlin",
                "dataset":"CIFAR10",
                "epochs":epochs,
                "optimizer":optimizer.__class__.__name__,
                "loss_fn":loss_fn.__class__.__name__,
                "blocks":num_blocks,
                "activation": args["activation"],
                "dropout":dropout,
                "batch_size":batch_size
            }
        )

    train_loader, valid_loader = get_data.get_data()
    trainer = Trainer(max_epochs = epochs, fast_dev_run=False)
    if torch.backends.mps.is_available():
        rainer = Trainer(max_epochs = epochs, fast_dev_run=False, accelerator="mps", devices=1)
    trainer.fit(model, train_loader, valid_loader)

    if track_wandb:
        wandb.finish()