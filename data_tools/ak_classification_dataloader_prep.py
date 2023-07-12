# this file creates the csv file that is used in ak_dataloader
# it creates a CSV file where the first row is the path to the image and the second row is the label

import csv
import json
from tqdm import tqdm
from PIL import Image

LABEL = "animal_parent_class"

annotation_dir = "datasets/Animal_Kingdom/pose_estimation/annotation/"
image_dir = "datasets/Animal_Kingdom/pose_estimation/dataset/"

target_category = "ak_P1"

row_list = [["image_directory", "label"]]

# do this for training data
f = open(f"{annotation_dir}{target_category}/train.json", "r")
annotations = json.load(f)
f.close()

for i in tqdm(range(len(annotations))):
    img = Image.open(f"{image_dir}{annotations[i]['image']}")
    width = img.width
    height = img.height
    if width == 640 and height == 360:
        row_list.append([annotations[i]["image"], annotations[i][LABEL]])

with open(f"data_tools/ak_classification_data_train.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

# do the same thing but for testing/validation data
f = open(f"{annotation_dir}{target_category}/test.json", "r")
annotations = json.load(f)
f.close()

for i in tqdm(range(len(annotations))):
    img = Image.open(f"{image_dir}{annotations[i]['image']}")
    width = img.width
    height = img.height
    if width == 640 and height == 360:
        row_list.append([annotations[i]["image"], annotations[i][LABEL]])

with open(f"data_tools/ak_classification_data_test.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(row_list)