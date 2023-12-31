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
from torchmetrics.classification import MultilabelF1Score

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
import argparse

cwd = os.path.dirname(os.path.realpath(__file__))
cwd = cwd[0:cwd.rfind("/")]
cwd = cwd[0:cwd.rfind("/") + 1]

sys.path.append(cwd)
from data_tools import ak_ar_images
from data_tools.ak_ar_images import dataloader

csv_path = cwd + "datasets/Animal_Kingdom/action_recognition/annotation/val.csv"

df = pd.read_excel(f"{cwd}datasets/Animal_Kingdom/action_recognition/annotation/df_action.xlsx")
landmarks_frame = pd.read_csv(csv_path, delimiter = " ")

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

class clip2_baseline(pl.LightningModule):
    def __init__(self, num_classes, threshold):
        super().__init__()

        # frozen model
        backbone = models.resnet50(weights="DEFAULT")
        backbone.fc = nn.Sequential(
            nn.Dropout(p = 0.2),
            nn.Linear(backbone.fc.in_features, num_classes * 16)
        )
        self.backbone = backbone

        # unfrozen model
        self.unfrozen = nn.Sequential(
            nn.Linear(num_classes * 16, num_classes * 16),
            nn.Dropout(p = 0.2),
            nn.Linear(num_classes * 16, num_classes * 4),
            nn.Dropout(p = 0.2),
            nn.Linear(num_classes * 4, num_classes)
        )
        self.num_classes = num_classes
        self.threshold = threshold

        self.sigm = nn.Sigmoid()

        self.train_step_losses = []
        self.validation_step_losses = []
        self.train_step_f1 = []
        self.validation_step_f1 = []
        self.train_step_acc = []
        self.validation_step_acc = []
        self.last_train_loss = 0
        self.last_train_f1 = 0
        self.last_train_acc = 0
    
    def forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(x)
        return self.sigm(self.unfrozen(x))

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # if torch.backends.mps.is_available():
        #     device = "mps"
        
        # x = self.backbone(images=x, return_tensors="pt").to(torch.float32).to(device)
        # # print(x["pixel_values"].shape)
        # x = x["pixel_values"].squeeze()
        # torch.reshape(x, (1, 128 * 3 * 224 * 224))
        # x = x.squeeze()
        # print(x.device)
        # return self.sigm(self.unfrozen(x))
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        # forward pass
        outputs = self(images)
        loss = nn.BCELoss()(outputs, labels.type(torch.float32))

        self.train_step_losses.append(loss)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"

        F1 = MultilabelF1Score(num_labels=self.num_classes, average="macro", threshold=self.threshold).to(device)
        self.train_step_f1.append(F1(outputs, labels.type(torch.float32)).item() * 100)

        outputs_clone = torch.clone(outputs).detach()
        N, C = labels.shape
        outputs_clone[outputs_clone >= self.threshold] = 1
        outputs_clone[outputs_clone < self.threshold] = 0
        accuracy = (outputs_clone == labels).sum() / (N*C) * 100
        self.train_step_acc.append(accuracy)

        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # forward pass
        outputs = self(images)
        loss = nn.BCELoss()(outputs, labels.type(torch.float32))

        self.validation_step_losses.append(loss)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"

        F1 = MultilabelF1Score(num_labels=self.num_classes, average="macro", threshold=self.threshold).to(device)
        self.validation_step_f1.append(F1(outputs, labels.type(torch.float32)).item() * 100)

        outputs_clone = torch.clone(outputs).detach()
        N, C = labels.shape
        outputs_clone[outputs_clone >= self.threshold] = 1
        outputs_clone[outputs_clone < self.threshold] = 0
        accuracy = (outputs_clone == labels).sum() / (N*C) * 100
        self.validation_step_acc.append(accuracy)

        return {'loss':loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), momentum=0.9, weight_decay=0.0001, lr = 0.001)
    
    def train_dataloader(self):
        train_loader, _ = dataloader.get_data()
        return train_loader
    
    def val_dataloader(self):
        _, val_loader = dataloader.get_data()
        return val_loader

    def on_train_epoch_end(self):
        self.last_train_loss = sum(self.train_step_losses) / len(self.train_step_losses)
        self.last_train_f1 = sum(self.train_step_f1) / len(self.train_step_f1)
        self.last_train_acc = sum(self.train_step_acc) / len(self.train_step_acc)

        # clear memory
        self.train_step_f1.clear()
        self.train_step_losses.clear()
        self.train_step_acc.clear()

    def on_validation_epoch_end(self):
        validation_loss = sum(self.validation_step_losses) / len(self.validation_step_losses)
        validation_f1 = sum(self.validation_step_f1) / len(self.validation_step_f1)
        validation_acc = sum(self.validation_step_acc) / len(self.validation_step_acc)

        wandb.log({"training_loss":self.last_train_loss,
                    "training_f1":self.last_train_f1,
                    "training_acc":self.last_train_acc,
                    "validation_loss":validation_loss,
                    "validation_f1":validation_f1,
                    "validation_acc":validation_acc})
        
        # clear memory
        self.validation_step_losses.clear()
        self.validation_step_f1.clear()
        self.validation_step_acc.clear()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--threshold', default='0.5',
        type = float,
        help='set the treshold of a positive prediction'
    )
    args = vars(parser.parse_args())

    f = open("data_tools/ak_ar_images/converted.json", "r")
    data = json.load(f)
    f.close()

    wandb.login()
    wandb.init(
        project="Perona_Research",
        config={
            "learning_rate":0.001,
            "architecture":"blip2_testing",
            "dataset":"ak_ar_image",
            "epochs":50,
            "optimizer":"adam",
            "loss_fn":"BCELoss",
            "dropout":0.2,
            "batch_size":128
        }
    )

    epochs = 50
    model = clip2_baseline(num_classes = len(data), threshold = args["threshold"])
    trainer = Trainer(max_epochs = epochs, fast_dev_run=False)
    trainer.fit(model)

    wandb.finish()