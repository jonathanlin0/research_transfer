# this runs single action recognition on the dataset

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


parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--data_split', default='all',
    type = str,
    required = False,
    help='set how the text labels are put into CLIP',
    choices=["all", "head", "middle", "tail", "ak_split"]
)
args = vars(parser.parse_args())
data_split = args["data_split"]


f = open("visualizer/data.json", "r")
data = json.load(f)
f.close()

# get the correct labels
df = pd.read_excel("datasets/Animal_Kingdom/action_recognition/annotation/df_action.xlsx")

if data_split != "all":
    if data_split == "head" or data_split == "middle" or data_split == "tail":
        correct_actions = []
        for index, row in df.iterrows():
            if row["segment"] == data_split:
                correct_actions.append(row["action"].lower())
        
        # remove actions from action["action_index_key"]
        to_remove = []
        for action in correct_actions:
            if action not in data["action_index_key"]:
                to_remove.append(action)
        for action in to_remove:
            correct_actions.remove(action)
        data["action_index_key"] = correct_actions

        # remove data from data["video_data"]
        data_temp = {}
        for vid_path in data["video_data"]:
            if data["video_data"][vid_path]["action"] in correct_actions:
                data_temp[vid_path] = data["video_data"][vid_path]
        
        data["video_data"] = data_temp
    elif data_split == "ak_split":
        # this is the split used in animal kingdom
        # the 5 animals are lizards, primates, spiders, orthopteran insects, water fowl
        # the namings arent exactly the same, because an animal has multiple types of names
        valid_actions = ["moving", "eating", "attending", "swimming", "sensing", "keeping still"]
        classes = ["lizard", "primate", "spider", "insect", "water bird"]
        modified_to_original = {}

        subclass_df = pd.read_excel("datasets/Animal_Kingdom/action_recognition/AR_metadata.xlsx", sheet_name="Animal")
        for index, row in subclass_df.iterrows():
            curr_classes = row["Sub-Class"].split(" / ")
            # turn all to lowercase
            for i in range(len(curr_classes)):
                curr_classes[i] = curr_classes[i].lower()

            matching_class = ""
            for curr_class in curr_classes:
                if curr_class in classes:
                    matching_class = curr_class
                    break
            
            if matching_class != "":
                for curr_class in curr_classes:
                    modified_to_original[curr_class] = matching_class
        
        # modify data obj to only have these 5 classes with the valid actions
        i = 0
        while i < len(data["action_index_key"]):
            if data["action_index_key"][i] not in valid_actions:
                data["action_index_key"].pop(i)
            else:
                i += 1
        
        data_copy = {}
        for data_pt in data["video_data"]:
            if data["video_data"][data_pt]["animal"].lower() in modified_to_original and data["video_data"][data_pt]["action"] in valid_actions:
                data_copy[data_pt] = data["video_data"][data_pt]
                data_copy[data_pt]["animal"] = modified_to_original[data["video_data"][data_pt]["animal"].lower()]
        
        data["video_data"] = data_copy

# ------------------------------------------------DATALOADER CLASS------------------------------------------------
class ak_ar_images_dataset(Dataset):
    def __init__(self, data_dir, annotation_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # convert the "data" variable to a list of tuples. first ele is the image, second ele is the label
        self.d = []

        csv_files = ["datasets/Animal_Kingdom/action_recognition/annotation/train.csv", "datasets/Animal_Kingdom/action_recognition/annotation/val.csv"]
        csv_files = [annotation_dir]
        df = pd.read_excel("datasets/Animal_Kingdom/action_recognition/annotation/df_action.xlsx")
        for file_path in csv_files:
            with open(file_path, "r") as csv_file:
                csvreader = csv.reader(csv_file, delimiter=" ")
                # skip the first row
                next(csvreader)
                for i, row in enumerate(csvreader):
                    video_name = row[0]
                    if video_name in data["video_data"]:
                        labels_index = int(row[4])
                        label = df.at[labels_index, "action"].lower()
                        image_path = row[3]
                        self.d.append((image_path, label))

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        # this dictionary converts the string labels to integers
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_dir + self.d[idx][0]
        label = self.d[idx][1]

        label_index = data["action_index_key"].index(label)
        # label_tensor = torch.tensor(label_index, dtype=torch.long)  # Convert to tensor

        image = PIL.Image.open(img_path, mode="r")
        if self.transform is not None:
            image = self.transform(image)
        image = image.to(torch.float32)

        return (image, label_index)


        return (image, label)

def get_data(batch_size=16, num_workers=8):
    # cwd = "/home/jonathan/Desktop/Perona_Research"
    cwd = os.path.dirname(os.path.realpath(__file__))
    cwd = cwd[0:cwd.rfind("/")]
    cwd = cwd[0:cwd.rfind("/") + 1]
    # set the root directory of the project to 2 layers above the current dataloader


    train_dataset = ak_ar_images_dataset(
        data_dir = "datasets/Animal_Kingdom/action_recognition/dataset/image/",
        annotation_dir = "datasets/Animal_Kingdom/action_recognition/annotation/train.csv",
        transform=torchvision.transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        # transforms.Resize((224, 224)),
                        transforms.RandAugment(),
                        transforms.ToTensor()
                     ])
    )

    val_dataset = ak_ar_images_dataset(
        data_dir = "datasets/Animal_Kingdom/action_recognition/dataset/image/",
        annotation_dir = "datasets/Animal_Kingdom/action_recognition/annotation/val.csv",
        transform=torchvision.transforms.Compose([
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
        self.dense = nn.Linear(num_filters, num_target_classes * 4)
        self.dense2 = nn.Linear(num_target_classes * 4, num_target_classes * 2)
        self.classifier = nn.Linear(num_target_classes * 2, num_target_classes)
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
    print(f"[INFO]: Set the data portion to {data_split}")

    track_wandb = False
    lr = 0.001
    num_classes = len(data["action_index_key"])
    epochs = 75
    dropout = 0.2

    if track_wandb:
        wandb.login()
        wandb.init(
            project="Perona_Research",
            config={
                "learning_rate": lr,
                "architecture": "calc_resnet",
                "dataset": "ak_ar_images",
                "epochs": epochs,
                "dropout": dropout,
                "batch_size": 64,
                "data_split": data_split
            }
        )

    model = calc_resnet(track_wandb=track_wandb,
                            lr=0.001,
                            num_classes=len(data["action_index_key"]),
                            dropout=0.2)
    
    trainer = Trainer(max_epochs = 50, fast_dev_run=False)
    trainer.fit(model)

    if track_wandb:
        wandb.finish()