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

class ak_ar_images_dataset(Dataset):
    """dataset for Animal Kingdom"""

    def __init__(self, csv_file, root_dir, total_classes, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.total_classes = total_classes
        

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # this dictionary converts the string labels to integers
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        
        label = self.landmarks_frame.iloc[idx, 1].split(",")
        converted = [eval(i) for i in label]
        # conver the list of indexes to a tensor of shape (total_classes) where the indexes are 1 (that are in converted car) and the rest are 0 in the tensor
        new_tensor = [0.0] * self.total_classes
        for index in converted:
            new_tensor[index] = 1.0
        label = torch.Tensor(new_tensor)

        image = PIL.Image.open(img_name, mode="r")
        # have to use PIL instead of io.imread because transform expects PIL image
        # image = io.imread(img_name)
        
        if self.transform is not None:
            image = self.transform(image)

        image = np.reshape(image, (3, 360, 640))
        image = image.to(torch.float32)

        return (image, label)

# cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = cwd[0:cwd.rfind("/")]
# cwd = cwd[0:cwd.rfind("/") + 1]
# val_dataset = ak_ar_images_dataset(
#         csv_file = cwd + "data_tools/ak_ar_images/val.csv",
#         root_dir = cwd + "datasets/Animal_Kingdom/action_recognition/dataset/image/",
#         total_classes = 139,
#         transform=torchvision.transforms.Compose([
#                         #  transforms.RandomHorizontalFlip(),
#                         #  transforms.RandAugment(),
#                          transforms.ToTensor()
#                      ])
#     )

# for i in range(len(val_dataset)):
#     print(val_dataset[i][1])

def get_data(batch_size=128, num_workers=8):
    # cwd = "/home/jonathan/Desktop/Perona_Research"
    cwd = os.path.dirname(os.path.realpath(__file__))
    cwd = cwd[0:cwd.rfind("/")]
    cwd = cwd[0:cwd.rfind("/") + 1]
    # set the root directory of the project to 2 layers above the current dataloader

    f = open(cwd + "data_tools/ak_ar_images/converted.json", "r")
    data = json.load(f)
    f.close()

    train_dataset = ak_ar_images_dataset(
        csv_file = cwd + "data_tools/ak_ar_images/train.csv",
        root_dir = cwd + "datasets/Animal_Kingdom/action_recognition/dataset/image/",
        total_classes = len(data),
        transform=torchvision.transforms.Compose([
                        #  transforms.RandomHorizontalFlip(),
                        #  transforms.RandAugment(),
                         transforms.ToTensor()
                     ])
    )

    val_dataset = ak_ar_images_dataset(
        csv_file = cwd + "data_tools/ak_ar_images/val.csv",
        root_dir = cwd + "datasets/Animal_Kingdom/action_recognition/dataset/image/",
        total_classes = len(data),
        transform=torchvision.transforms.Compose([
                        #  transforms.RandomHorizontalFlip(),
                        #  transforms.RandAugment(),
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