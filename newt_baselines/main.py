import csv
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
import PIL
from PIL import Image
import cv2
import os

BATCH_SIZE = 16
IMAGE_DIR = "datasets/newt/newt2021_images/"
LABELS_DIR = "datasets/newt/newt2021_labels.csv"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

parser = argparse.ArgumentParser()
parser.add_argument(
    '-g', '--granularity', default='nothing',
    type = str,
    required = False,
    help='set the granularity of the CLIP model',
    choices=["nothing", "animal"]
)
args = vars(parser.parse_args())
args_granularity = args["granularity"]
print(f"[INFO]: Set the granularity to {args_granularity}")

# read in csv files
f = open("newt_baselines/prep_data.json", "r")
data = json.load(f)
train, val = data["train"], data["val"]
f.close()
data = train + val

sizes = set()
for img in data:
    image = PIL.Image.open(img[0], mode="r")
    sizes.add(image.size)

unique_labels = set()
with open(LABELS_DIR, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    
    for row in reader:
        file_path = row[0]
        task = row[1]
        subtask = row[3]
        text = row[5]
        if task == "behavior":
            unique_labels.add(text)
unique_labels = list(unique_labels)

# generate the labels
# value is the label inputted into CLIP, and the original label
label_to_original = {}

if args_granularity == "nothing":
    for label in unique_labels:
        label_to_original[label] = label
elif args_granularity == "animal":
    for label in unique_labels:
        label_to_original[f"an animal is {label}"] = label

index_to_label = {}
for i in range(len(label_to_original.keys())):
    index_to_label[i] = list(label_to_original.keys())[i]

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

i = 0
all_labels_true = []
all_labels_pred = []
# with tqdm(total=len(data)) as pbar:
while i < len(data):
    # pbar.update(i)
    # add up to batch_size number of images
    images = []
    labels_true = []
    labels_pred = []
    j = i
    while len(images) < BATCH_SIZE and j < len(data):
        images.append(Image.open(data[j][0]))
        labels_true.append(data[j][1])
        j += 1
    i = max(i + 1, j)
    
    # images = np.array(images)
    images = np.stack(images)

    inputs = processor(
        text=list(label_to_original.keys()),
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    for k, subarray in enumerate(logits_per_image):
        subarray.softmax(dim=0)
        curr_label = label_to_original[index_to_label[torch.argmax(subarray.cpu(), dim=(0)).item()]]
        labels_pred.append(curr_label)
    
    all_labels_true += labels_true
    all_labels_pred += labels_pred

cnt = 0
for i in range(len(all_labels_true)):
    if all_labels_true[i] == all_labels_pred[i]:
        cnt += 1

print(f"Accuracy: {cnt / len(all_labels_true)}")

existing_data = {}
try:
    f = open("newt_baselines/run_data.json", "r")
    existing_data = json.load(f)
    f.close()
except:
    pass
existing_data[args_granularity] = {"acc": cnt / len(all_labels_true), "true": all_labels_true, "pred": all_labels_pred}
with open("newt_baselines/run_data.json", "w") as f:
    json.dump(existing_data, f, indent=4)




# calculate using batches of batch_size
# correct data