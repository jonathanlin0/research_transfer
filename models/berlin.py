from typing import Any, Type

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
import wandb

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from torchmetrics import Accuracy
import torch.utils.data as data
from torchvision.datasets import MNIST
from torch.nn import functional as F

from data_tools import get_data

import sys
sys.path.append("..")

class Block_Berlin(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super(Block_Berlin, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm_1 = nn.BatchNorm2d(self.out_channels)
        self.activation_1 = activation
        
        self.conv_2 = nn.Conv2d(
            in_channels = self.out_channels,
            out_channels = self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm_2 = nn.BatchNorm2d(self.out_channels)
        self.activation_2 = activation

        # downsample adjusts the original input to fit the shape
        self.downsample = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = 1,
            stride = 2,
            padding = 0,
            bias=False
        )

        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.1)

    def forward(self, x):
        identity = x

        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation_1(x)
        # x = self.dropout_1(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        # x = self.dropout_2(x)

        # apply identity function
        x = x + self.downsample(identity)
        x = self.activation_2(x)

        return x  
    
class berlin(pl.LightningModule):
    def __init__(
        self,
        img_channels: int,
        block: Type[Block_Berlin],
        num_blocks: int,
        activation,
        lr: float,
        dropout: float,
        num_classes: int,
        track_wandb:bool):
        super(berlin, self).__init__()

        self.prep = nn.Conv2d(
                in_channels = img_channels,
                out_channels = 64,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                bias = False
        )
        self.norm_1 = nn.BatchNorm2d(64)
        self.activation = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        layers = []
        current_in = 64
        # create number of blocks according to the num_blocks input
        for i in range(1, num_blocks + 1):
            layers.append(block(current_in, current_in * 2, activation = activation))
            current_in *= 2
        
        self.middle_section = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * (2 ** num_blocks), num_classes)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

        self.lr = lr
        self.track_wandb = track_wandb
        self.train_step_losses = []
        self.validation_step_losses = []
        self.train_step_acc = []
        self.validation_step_acc = []
        self.last_train_acc = 0
        self.last_train_loss = 0
    
    def forward(self, x):
        x = self.prep(x)
        x = self.norm_1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.dropout_1(x)

        x = self.middle_section(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout_2(x)

        return x
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        # images = images.reshape(-1, 32 * 32)

        # forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.train_step_losses.append(loss)
        
        # log accuracy
        _,preds = torch.max(outputs.data, 1)
        acc = (preds == labels).sum().item()
        acc /= outputs.size(dim=0)
        acc *= 100
        self.train_step_acc.append(acc)

        return {'loss':loss}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    def train_dataloader(self):
        train_loader, val_loader = get_data.get_data()
        return train_loader
    
    def on_train_epoch_end(self):
        all_preds = self.train_step_losses
        avg_loss = sum(all_preds) / len(all_preds)
        
        all_acc = self.train_step_acc
        avg_acc = sum(all_acc) / len(all_acc)
        avg_acc = round(avg_acc, 2)

        self.last_train_acc = avg_acc
        self.last_train_loss = avg_loss

        # clear memory
        self.train_step_acc.clear()
        self.train_step_losses.clear()
        
        return {'train_loss':avg_loss}
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # images = images.reshape(-1, 28 * 28)
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.validation_step_losses.append(loss.item())

        # log accuracy
        _,preds = torch.max(outputs.data, 1)
        acc = (preds == labels).sum().item()
        acc /= outputs.size(dim=0)
        acc *= 100
        self.validation_step_acc.append(acc)

        return {'val_loss':loss}
    
    def val_dataloader(self):
        train_loader, val_loader = get_data.get_data()
        return val_loader
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.l1(x)
        x_hat = self.l2(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)
    
    def on_validation_epoch_end(self):
        all_preds = self.validation_step_losses
        all_acc = self.validation_step_acc

        avg_loss = sum(all_preds) / len(all_preds)
        avg_acc = sum(all_acc) / len(all_acc)
        avg_acc = round(avg_acc, 2)

        if self.track_wandb:
            wandb.log({"training_loss":self.last_train_loss,
                        "training_acc":self.last_train_acc,
                        "validation_loss":avg_loss,
                        "validation_acc":avg_acc})

        # clear memory
        self.validation_step_losses.clear()
        self.validation_step_acc.clear()

        return {'val_loss':avg_loss}