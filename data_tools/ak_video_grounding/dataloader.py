import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

# the structure of this dataset will follow CIFAR10 where __getitem__ returns a tuple of (image, target) where target is the index of the target class
class ak_vg_dataset(Dataset):
    """dataset for Animal Kingdom"""

    def __init__(self, csv_file, root_dir, animal_label, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def read_video_cv2(self, video_path, n_frames=100):
        cap = cv2.VideoCapture(video_path)
        all = []
        i = 0
        while cap.isOpened() and i < n_frames:
            ret, frame = cap.read()
            arr = np.array(frame)
            all.append(arr)
            i += 1
        return np.array(all)

    def __getitem__(self, idx):
        # this dictionary converts the string labels to integers
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        
        start_time = self.landmarks_frame.iloc[idx, 1]
        end_time = self.landmarks_frame.iloc[idx, 2]
        caption = self.landmarks_frame.iloc[idx, 3]

        # multiply by 30 since the videos are 30 fps
        video = self.read_video_cv2(video_path, (end_time - start_time) * 30)

        # image = io.imread(img_name)
        # image = np.reshape(image, (3, 360, 640))
        # video = (torch.from_numpy(video)).to(torch.float32)

        return (image, video, caption)



# def get_data(batch_size=100, num_workers=8):
#     # cwd = "/home/jonathan/Desktop/Perona_Research"
#     cwd = "/Users/jonathanlin/Documents/GitHub/research_transfer/"
#     # for the lab computer directory
#     if torch.cuda.is_available():
#         cwd = "/home/jonathan/Desktop/Perona_Research/"

#     train_dataset = ak_classification_dataset(
#         csv_file = cwd + "data_tools/ak_classification/dataset_train.csv",
#         root_dir = cwd + "datasets/Animal_Kingdom/pose_estimation/dataset/",
#         animal_label = "animal_parent_class"
#     )

#     val_dataset = ak_classification_dataset(
#         csv_file = cwd + "data_tools/ak_classification/dataset_test.csv",
#         root_dir = cwd + "datasets/Animal_Kingdom/pose_estimation/dataset/",
#         animal_label = "animal_parent_class"
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size = batch_size,
#         num_workers = num_workers,
#         shuffle = True
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size = batch_size,
#         num_workers = num_workers,
#         shuffle = False
#     )

#     return train_loader, val_loader