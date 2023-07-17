cwd = "/Users/jonathanlin/Documents/GitHub/research_transfer/"
csv_path = cwd + "datasets/Animal_Kingdom/action_recognition/annotation/val.csv"

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
import json
sys.path.append("/Users/jonathanlin/Documents/GitHub/research_transfer")
from data_tools import ak_ar_images
from data_tools.ak_ar_images import dataloader

df = pd.read_excel("/Users/jonathanlin/Documents/GitHub/research_transfer/datasets/Animal_Kingdom/action_recognition/annotation/df_action.xlsx")
landmarks_frame = pd.read_csv(csv_path, delimiter = " ")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# caption_model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32
# )

class clip2_baseline(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        # frozen model
        self.backbone = processor

        # unfrozen model
        self.unfrozen = nn.Sequential(
            nn.Linear(3 * 224 * 224, num_classes * 4),
            nn.Dropout(p = 0.2),
            nn.Linear(num_classes * 4, num_classes)
        )

        self.sigm = nn.Sigmoid()
    
    def forward(self, x):
        # self.backbone.eval()
        # with torch.no_grad():
        output = self.backbone(images=x, return_tensors="pt").to(torch.float32)["pixel_values"]
        # output = torch.squeeze(output["pixel_values"])
        output = output.type_as(x)
        x = torch.squeeze(output)
        
        # return self.sigm(self.unfrozen(output))
        return self.sigm(self.unfrozen(x.view(x.size(0), -1)))
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        # images = images.reshape(-1, 32 * 32)

        # forward pass
        outputs = self(images)
        # loss = nn.BCEWithLogitsLoss()
        loss = nn.BCELoss()(outputs, labels.type(torch.float32))

        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # images = images.reshape(-1, 32 * 32)

        # forward pass
        outputs = self(images)
        loss = nn.BCELoss()(outputs, labels.type(torch.float32))

        return {'loss':loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 0.001)
    
    def train_dataloader(self):
        train_loader, val_loader = dataloader.get_data()
        return train_loader
    
    def val_dataloader(self):
        train_loader, val_loader = dataloader.get_data()
        return val_loader

if __name__ == "__main__":

    # f = open("/Users/jonathanlin/Documents/GitHub/research_transfer/data_tools/ak_ar_images/converted.json", "r")
    # data = json.load(f)
    # f.close()

    # train_loader, val_loader = dataloader.get_data()

    epochs = 5
    model = clip2_baseline(num_classes = 46)
    trainer = Trainer(max_epochs = epochs, fast_dev_run=False)
    trainer.fit(model)

    trainer.save_checkpoint("temp.ckpt")