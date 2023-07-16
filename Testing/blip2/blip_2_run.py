import pandas as pd
import PIL
import random
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
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
import os
import sys

cwd = os.path.dirname(os.path.realpath(__file__))
cwd = cwd[0:cwd.rfind("/")]
cwd = cwd[0:cwd.rfind("/") + 1]

sys.path.append(cwd)
from data_tools import ak_ar_images
from data_tools.ak_ar_images import dataloader

csv_path = cwd + "datasets/Animal_Kingdom/action_recognition/annotation/val.csv"

df = pd.read_excel(f"{cwd}datasets/Animal_Kingdom/action_recognition/annotation/df_action.xlsx")
landmarks_frame = pd.read_csv(csv_path, delimiter = " ")

class clip2_baseline(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        # frozen model
        backbone = models.resnet50(weights="DEFAULT")
        backbone.fc = nn.Sequential(
            nn.Dropout(p = 0.2),
            nn.Linear(backbone.fc.in_features, out_features=num_classes)
        )
        self.backbone = backbone

        self.sigm = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigm(self.backbone(x))
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        # images = images.reshape(-1, 32 * 32)

        # forward pass
        outputs = self(images)
        # loss = nn.BCEWithLogitsLoss()
        loss = nn.BCEWithLogitsLoss()(outputs, labels.type(torch.float32))

        print(loss)
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # images = images.reshape(-1, 32 * 32)

        # forward pass
        outputs = self(images)
        loss = nn.BCEWithLogitsLoss()(outputs, labels.type(torch.float32))

        return {'loss':loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 0.001)
    
    def train_dataloader(self):
        train_loader, val_loader = dataloader.get_data()
        return train_loader
    
    def val_dataloader(self):
        train_loader, val_loader = dataloader.get_data()
        return val_loader
    
if __name__ == '__main__':
    epochs = 10
    model = clip2_baseline(num_classes = 139)
    trainer = Trainer(max_epochs = epochs, fast_dev_run=False)
    trainer.fit(model)