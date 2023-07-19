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
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from data_tools import ak_classification
from data_tools.ak_classification import dataloader

import ssl
import urllib.request

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

class cairo(pl.LightningModule):
    def __init__(self, 
                 num_classes: int,
                 is_blip2_modal: bool = True,
                 is_blip2_processing_modal: bool = False,
                 is_caption_modal: bool = True,
                 blip2_width: int = 224,
                 blip2_height: int = 224,
                 blip2_color_channels: int = 3,
                 caption_embed_size: int = 384,
                 lr: float = 0.001,
                 dropout: float = 0.2,
                 track_wandb: bool = False):
        super().__init__()

        self.is_blip2_modal = is_blip2_modal
        self.is_blip2_processing_modal = is_blip2_processing_modal
        self.is_caption_modal = is_caption_modal
        self.blip2_width = blip2_width
        self.blip2_height = blip2_height
        self.blip2_color_channels = blip2_color_channels
        self.caption_embed_size = caption_embed_size

        if is_blip2_modal:
            self.blip2_modal = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if is_blip2_processing_modal:
            if is_blip2_modal == False:
                self.blip2_modal = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip2_processor_modal = models.resnet50(weights="DEFAULT")
        if is_caption_modal:
            self.caption_modal = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32
            )
            self.tokenizer = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        
        assert (is_blip2_modal or is_caption_modal)

        # set the input dimensions to the unfrozen layer
        input_size = 0
        if is_blip2_modal:
            input_size += (blip2_width * blip2_height * blip2_color_channels)
        if is_caption_modal:
            input_size += (caption_embed_size)
        self.input_size = input_size
        
        # unfrozen model
        self.unfrozen = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Dropout(p = dropout),
            nn.Linear(input_size // 2, input_size // 4),
            nn.Dropout(p = dropout),
            nn.Linear(input_size // 4, num_classes)
        )

        self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        # self.backbone.eval()
        # with torch.no_grad():
        output = self.blip2_modal(images=x, return_tensors="pt").to(torch.float32)["pixel_values"]

        # master input is what goes into the unfrozen layer
        master_input = None

        if self.is_blip2_modal:
            master_input = output.type_as(x)
            master_input = torch.squeeze(master_input)
            master_input = torch.squeeze(master_input)
            master_input = master_input.view(master_input.size(0), -1)

        if self.is_caption_modal:
            generated_ids = self.caption_modal.generate(**output)
            generated_text = self.caption_modal.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            embedding = self.tokenizer(generated_text)
            embedding = embedding.squeeze()
            if master_input != None:
                master_input = torch.cat((master_input, embedding), 0)
            else:
                master_input = embedding
        
        # return self.sigm(self.unfrozen(output))
        return self.sigm(self.unfrozen(master_input))
    
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
        train_loader, val_loader = dataloader.get_data(batch_size=16)
        return train_loader
    
    def val_dataloader(self):
        train_loader, val_loader = dataloader.get_data(batch_size=16)
        return val_loader

if __name__ == "__main__":

    # f = open("/Users/jonathanlin/Documents/GitHub/research_transfer/data_tools/ak_ar_images/converted.json", "r")
    # data = json.load(f)
    # f.close()

    # train_loader, val_loader = dataloader.get_data()

    epochs = 5
    model = cairo(num_classes = 46)
    trainer = Trainer(max_epochs = epochs, fast_dev_run=False)
    trainer.fit(model)

    # trainer.save_checkpoint("temp.ckpt")