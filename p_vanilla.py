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

# import the dif models and parts
from models import vanilla_beijing
from models.vanilla_beijing import Block_Beijing
from models import vanilla_berlin
from models.vanilla_berlin import Block_Berlin

# import utility functions
from data_tools import get_data

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

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(model: torch.nn.Module,
               data_loader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device=device):
    train_loss, train_acc = 0,0

    model.train()
    
    for batch, (X, y) in tqdm(enumerate(data_loader), total=len(data_loader)):
        torch.set_grad_enabled(True)
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # train_acc += accuracy_fn(y_true=y.cpu(),
        #                         y_pred=y_pred.argmax(dim=1).cpu()) # go from logits -> prediction labels
        _,preds = torch.max(y_pred.data, 1)
        train_acc += (preds == y).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(data_loader)
    train_acc /= len(data_loader.dataset)
    train_acc *= 100
    print(f"train loss: {train_loss:.3f} | train acc: {train_acc:.2f}%")
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
            data_loader,
            loss_fn : torch.nn.Module,
            accuracy_fn,
            device: torch.device=device):
    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for i, (X, y) in tqdm(enumerate(data_loader), total=len(data_loader)):
        #for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)

            #test_acc += accuracy_fn(y_true=y.cpu().numpy(), y_pred=test_pred.argmax(dim=1).cpu().numpy())
            _,preds = torch.max(test_pred.data, 1)
            test_acc += (preds == y).sum().item()
            #calculate test loss avg per batch
        test_loss /= len(data_loader)
        test_acc /= len(data_loader.dataset)
        test_acc *= 100
    print(f"test loss: {test_loss:.3f} | test acc {test_acc:.2f}%")
    return test_loss, test_acc

torch.manual_seed(42)
torch.cuda.manual_seed(42)

track_wandb = False

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

# model_0 = beijing.beijing(img_channels=3,
#                   block=Block_Beijing,
#                   num_blocks = num_blocks,
#                   activation = activation,
#                   num_classes=10).to(device)
model_0 = vanilla_berlin.berlin(img_channels=3,
                  block=Block_Berlin,
                  num_blocks = num_blocks,
                  activation = activation,
                  dropout = dropout,
                  num_classes=10).to(device)

learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(),
                            lr=learning_rate)

# define model based on the argument parser string
if args["learning_rate"] == "0.1":
    print("[INFO]: Set learning rate to 0.1")
    learning_rate = 0.1
if args["learning_rate"] == "0.01":
    print("[INFO]: Set learning rate to 0.01")
    learning_rate = 0.01
if args["learning_rate"] == "0.001":
    print("[INFO]: Set learning rate to 0.001")
    learning_rate = 0.001
if args["optimizer"] == "Adam":
    print("[INFO]: Set optimizer to Adam")
    optimizer = torch.optim.Adam(params = model_0.parameters(),
                                 lr = learning_rate)
if args["optimizer"] == "SGD":
    print("[INFO]: Set optimizer to SGD")
    optimizer = torch.optim.SGD(params = model_0.parameters(),
                                lr = learning_rate)


train_dataloader, val_dataloader = get_data.get_data(batch_size)


for param in model_0.parameters():
    param.requires_grad = True

model_0.eval()
epochs = 200

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

for epoch in range(epochs):
    print("\n")
    print(f"epoch {epoch + 1} out of {epochs}")
    train_loss, train_acc = train_step(model=model_0,
                data_loader=train_dataloader,
                loss_fn = loss_fn,
                optimizer=optimizer,
                accuracy_fn = accuracy_score,
                device=device)
    test_loss, test_acc = test_step(model=model_0,
                data_loader=val_dataloader,
                loss_fn = loss_fn,
                accuracy_fn = accuracy_score,
                device=device)
    if track_wandb:
        wandb.log({"training_loss":train_loss,
                    "training_acc":train_acc,
                    "validation_loss":test_loss,
                    "validation_acc":test_acc})

if track_wandb:
    wandb.finish()