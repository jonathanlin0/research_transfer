# this file does action recognition on the new newt dataset

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import torchvision
import PIL
from transformers import AutoProcessor, AutoModel
from contextlib import redirect_stdout, redirect_stderr
import json
import argparse
from transformers import CLIPProcessor, CLIPModel
import av
import copy
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image
import cv2
import csv
import os
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
from torchvision.datasets import MNIST
from torch.nn import functional as F
import torchvision.models as models
import random

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights_and_biases', default=False,
    type = bool,
    required = False,
    help='set whether to track the model w wandb',
    choices=[True, False]
)
parser.add_argument(
    '-b', '--batch_size', default=16,
    type = int,
    required = False,
    help='set the batch size',
    choices=[4,8,16,32,64,128,256,512,1024]
)
parser.add_argument(
    '-p', '--val_split_probability', default=0.2,
    type = float,
    required = False,
    help='set the percentage of the dataset that is used for validation',
)
args = vars(parser.parse_args())
track_wandb = args["weights_and_biases"]
arg_batch_size = args["batch_size"]
val_split_prob = args["val_split_probability"]

# key is the video name. value is the label (in string form)
data_train = {}
data_val = {}

# read in the data
with open('datasets/newt/newt2021_labels.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Skip the header row
    next(csv_reader)

    # Iterate through the rows
    for row in csv_reader:
        id = row[0]
        cluster = row[1]
        if cluster == "behavior":
            label = row[5]
            num = random.random()
            if num <= val_split_prob:
                data_val[id + ".jpg"] = label
            else:
                data_train[id + ".jpg"] = label
            
# convert the labels to integers
all_labels = list(set(list(data_train.values()) + list(data_val.values())))
string_to_int_labels = {}
for i, label in enumerate(all_labels):
    string_to_int_labels[label] = i

for key in data_train:
    data_train[key] = string_to_int_labels[data_train[key]]
for key in data_val:
    data_val[key] = string_to_int_labels[data_val[key]]

heights = []
widths = []

# look at the shape of the images
for key in list(data_train.keys()) + list(data_val.keys()):
    from PIL import Image

    # Open an image
    image = Image.open("datasets/newt/newt2021_images/" + key)

    # Get the dimensions (width and height)
    width, height = image.size

    heights.append(height)
    widths.append(width)

avg_height = (sum(heights) / len(heights))
avg_width = (sum(widths) / len(widths))



# ------------------------------------------------DATALOADER CLASS------------------------------------------------
class ak_ar_images_dataset(Dataset):
    def __init__(self, dataset_type, data_dir="datasets/newt/newt2021_images/", transform=None):

        valid_types = {"train", "val"}
        if dataset_type not in valid_types:
            raise ValueError("results: status must be one of %r." % valid_types)

        self.data_dir = data_dir
        self.transform = transform
    
        images = data_train if dataset_type == "train" else data_val

        # convert the "data" variable to a list of tuples. first ele is the image, second ele is the label
        self.d = list(images.items())

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        # this dictionary converts the string labels to integers
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_dir + self.d[idx][0]
        label = self.d[idx][1]

        image = PIL.Image.open(img_path, mode="r")
        if self.transform is not None:
            image = self.transform(image)
        image = image.to(torch.float32)

        return (image, label)


def get_data(batch_size=arg_batch_size, num_workers=8):
    # cwd = "/home/jonathan/Desktop/Perona_Research"
    cwd = os.path.dirname(os.path.realpath(__file__))
    cwd = cwd[0:cwd.rfind("/")]
    cwd = cwd[0:cwd.rfind("/") + 1]
    # set the root directory of the project to 2 layers above the current dataloader


    train_dataset = ak_ar_images_dataset(
        dataset_type = "train",
        transform=torchvision.transforms.Compose([
                        transforms.CenterCrop((int(avg_height), int(avg_width))),
                        transforms.RandomHorizontalFlip(),
                        # transforms.Resize((224, 224)),
                        transforms.RandAugment(),
                        transforms.ToTensor()
                     ])
    )

    val_dataset = ak_ar_images_dataset(
        dataset_type = "val",
        transform=torchvision.transforms.Compose([
                        transforms.CenterCrop((int(avg_height), int(avg_width))),
                        #  transforms.RandomHorizontalFlip(),
                        #  transforms.RandAugment(),
                        # transforms.Resize((224, 224)),
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


class calc_resnet(pl.LightningModule):
    def __init__(self, 
        track_wandb: bool,
        lr: float,
        num_classes: int,
        dropout: float):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = num_classes
        self.dense = nn.Linear(num_filters, num_filters // 2)
        self.dense2 = nn.Linear(num_filters // 2, num_filters // 4)
        self.dense3 = nn.Linear(num_filters // 4, num_filters // 8)
        self.dense4 = nn.Linear(num_filters // 8, num_filters // 16)
        self.classifier = nn.Linear(num_filters // 16, num_target_classes)
        self.dropout = nn.Dropout(dropout)

        self.lr = lr
        self.track_wandb = track_wandb
        self.train_step_losses = []
        self.validation_step_losses = []
        self.train_step_acc = []
        self.validation_step_acc = []
        self.last_train_acc = 0
        self.last_train_loss = 0

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        
        x = self.dropout(representations)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = self.dropout(x)
        x = self.dense4(x)
        x = self.dropout(x)
        x = self.classifier(x)

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
        train_loader, val_loader = get_data()
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
        train_loader, val_loader = get_data()
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

# ------------------------------------------------MAIN------------------------------------------------
if __name__ == "__main__":
    print(f"[INFO]: Set the wandb tracking to {track_wandb}")
    print(f"[INFO]: Set the batch size to {arg_batch_size}")
    print(f"[INFO]: Set the validation split probability to {val_split_prob}")

    lr = 0.001
    num_classes = len(all_labels)
    epochs = 75
    dropout = 0.2

    if track_wandb:
        wandb.login()
        wandb.init(
            project="Perona_Research",
            config={
                "learning_rate": lr,
                "architecture": "calc_resnet",
                "dataset": "newt_ar_images",
                "epochs": epochs,
                "dropout": dropout,
                "batch_size": arg_batch_size,
                "val_split_probability": val_split_prob,
            }
        )

    model = calc_resnet(track_wandb=track_wandb,
                            lr=0.001,
                            num_classes=num_classes,
                            dropout=0.2)
    
    trainer = Trainer(max_epochs = 50, fast_dev_run=False)
    trainer.fit(model)

    if track_wandb:
        wandb.finish()