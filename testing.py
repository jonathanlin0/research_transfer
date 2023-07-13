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
import torchvision.models as models

from data_tools import ak_classification
from data_tools.ak_classification import dataloader

import ssl
import urllib.request

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

import sys
sys.path.append("..")
    
class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 5
        self.classifier = nn.Linear(num_filters, num_target_classes)

        self.lr = 0.001

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
    
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
        train_loader, val_loader = dataloader.get_data()
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
        train_loader, val_loader = dataloader.get_data()
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
        
        print(self.last_train_acc)
        print(avg_acc)

        # clear memory
        self.validation_step_losses.clear()
        self.validation_step_acc.clear()

        return {'val_loss':avg_loss}

if __name__ == "__main__":
    model = ImagenetTransferLearning()
    trainer = Trainer()
    trainer.fit(model)

