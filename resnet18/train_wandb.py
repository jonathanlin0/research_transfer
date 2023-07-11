import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
import wandb

from resnet18 import ResNet, BasicBlock
from resnet18_torchvision import build_model
from training_utils import train, validate
from utils import save_plots, get_data

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='scratch',
    help='choose model built from scratch or the Torchvision model',
    choices=['scratch', 'torchvision']
)
parser.add_argument(
    '-l', '--learning_rate', default='0.01',
    help='set the learning rate of the optimizer',
    choices=['0.1', '0.01', '0.001']
)
args = vars(parser.parse_args())

# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

epochs = 200
batch_size = 64
learning_rate = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, valid_loader = get_data(batch_size=batch_size)

# define model based on the argument parser string
if args["model"] == "scratch":
    print("[INFO]: Training ResNet18 built from scratch...")
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
    plot_name = "resnet_scratch"
if args["model"] == "torchvision":
    print("[INFO]: Training the TorchVision ResNet18 Model...")
    model = build_model(pretrained=False, fine_tune=True, num_classes=10).to(device)
    plot_name = "resnet_torchvision"
if args["learning_rate"] == "0.1":
    print("[INFO]: Set learning rate to 0.1")
    learning_rate = 0.1
if args["learning_rate"] == "0.01":
    print("[INFO]: Set learning rate to 0.01")
    learning_rate = 0.01
if args["learning_rate"] == "0.001":
    print("[INFO]: Set learning rate to 0.001")
    learning_rate = 0.001

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
# Optimizer.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Loss function.
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # wandb stuff
    wandb.init(
        project="Perona_Research",

        config={
            "learning_rate":learning_rate,
            "architecture":"ResNet18",
            "dataset":"CIFAR10",
            "epochs":epochs,
            "optimizer":optimizer.__class__.__name__,
            "loss_fn":criterion.__class__.__name__,
        }
    )

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, 
            train_loader, 
            optimizer, 
            criterion,
            device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, 
            valid_loader, 
            criterion,
            device
        )

        # log on wandb
        wandb.log({"training_loss":train_epoch_loss,
                    "training_acc":train_epoch_acc,
                    "validation_loss":valid_epoch_loss,
                    "validation_acc":valid_epoch_acc})

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        
    # Save the loss and accuracy plots.
    save_plots(
        train_acc, 
        valid_acc, 
        train_loss, 
        valid_loss, 
        name=plot_name
    )
    print('TRAINING COMPLETE')

    wandb.finish()