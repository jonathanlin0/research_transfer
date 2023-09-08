# this reshapes all the images to a set dimension

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
from PIL import Image
from math import floor
from math import ceil
from statistics import median
import shutil
from tqdm import tqdm
import random
import csv

VAL_SPLIT_PROBS = 0.2
IMAGE_DIR = "datasets/newt/newt2021_images/"
LABELS_DIR = "datasets/newt/newt2021_labels.csv"
train = []
val = []
with open(LABELS_DIR, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    
    for row in reader:
        file_path = row[0]
        task = row[1]
        subtask = row[3]
        text = row[5]
        if task == "behavior":
            if random.random() <= VAL_SPLIT_PROBS:
                val.append([IMAGE_DIR + file_path + ".jpg", text])
            else:
                train.append([IMAGE_DIR + file_path + ".jpg", text])

# delete all contents in cache_photos
try:
    shutil.rmtree('newt_baselines/cache_photos')
except:
    print("cache_photos folder does not exist. Creating new one...")
os.mkdir("newt_baselines/cache_photos")

dicts = [train, val]

height = 480
width = 640
new_train = []
new_val = []
matching_dicts = [new_train, new_val]
for i, dict_ in enumerate(dicts):
    for j, key in enumerate(tqdm(dict_)):
        image_path = key[0]
        image = PIL.Image.open(image_path, mode="r")

        # remove grayscale images
        if image.size[0] == 1:
            continue

        # if image is extremely long, cut the image left and right sides
        # if the image is extremely tall, cut the image top and bottom sides
        curr_width = image.size[0]
        curr_height = image.size[1]
        width_ratio = width / curr_width
        height_ratio = height / curr_height
        # shrink the image so that the dimension that has a lower ratio is the same as the target dimension
        if curr_width >= width and curr_height >= height:
            if width_ratio > height_ratio:
                shrink_ratio = height_ratio
                
                # resize image while maintaining ratio
                target_width = round(curr_width * shrink_ratio)
                target_height = round(curr_height * shrink_ratio)
                image = image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)

                # cut off the edges of width
                curr_width = image.size[0]
                left_cutoff = floor((curr_width - width) / 2)
                right_cutoff = ceil((curr_width - width) / 2)
                image = image.crop((left_cutoff, 0, curr_width - right_cutoff, height))
            else:
                shrink_ratio = width_ratio

                # resize image while maintaining ratio
                target_width = round(curr_width * shrink_ratio)
                target_height = round(curr_height * shrink_ratio)
                image = image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)

                # cut off the edges of height
                curr_height = image.size[1]
                top_cutoff = floor((curr_height - height) / 2)
                bottom_cutoff = ceil((curr_height - height) / 2)
                image = image.crop((0, top_cutoff, width, curr_height - bottom_cutoff))
        # one of the dimensions is too small
        # scale the image (keeping aspect ratio) so that the smaller dimension is the same as the target dimension
        elif height_ratio > width_ratio:
            # upscale for height to match
            scale_ratio = height / curr_height
            target_width = round(curr_width * scale_ratio)
            target_height = round(curr_height * scale_ratio)
            image = image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)

            # cut off the edges of width
            curr_width = image.size[0]
            left_cutoff = floor((curr_width - width) / 2)
            right_cutoff = ceil((curr_width - width) / 2)
            image = image.crop((left_cutoff, 0, curr_width - right_cutoff, height))
        else:
            # upscale for width to match
            scale_ratio = width / curr_width
            target_width = round(curr_width * scale_ratio)
            target_height = round(curr_height * scale_ratio)
            image = image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)

            # cut off the edges of height
            curr_height = image.size[1]
            top_cutoff = floor((curr_height - height) / 2)
            bottom_cutoff = ceil((curr_height - height) / 2)
            image = image.crop((0, top_cutoff, width, curr_height - bottom_cutoff))
        
        image_path = image_path.replace("/", "_")
        image_path = image_path[:image_path.rfind(".")] # get rid of current file extension
        image_path = "newt_baselines/cache_photos/" + image_path + ".png"
        
        matching_dicts[i].append([image_path, key[1]])
        image.save(image_path ,"PNG")

train = new_train
val = new_val

# save the json file
with open("newt_baselines/prep_data.json", "w") as f:
    json.dump({"train": train, "val": val}, f, indent=4)