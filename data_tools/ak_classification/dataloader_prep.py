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

dimensions = {}

classes_list = []
classes_set = set()

# do this for training data
f = open(f"{annotation_dir}{target_category}/train.json", "r")
annotations = json.load(f)
f.close()

for i in tqdm(range(len(annotations))):
    img = Image.open(f"{image_dir}{annotations[i]['image']}")
    width = img.width
    height = img.height
    if (height, width) in dimensions:
        dimensions[(height, width)] += 1
    else:
        dimensions[(height, width)] = 1
    if width == 640 and height == 360:
        curr_class = annotations[i][LABEL]
        if curr_class not in classes_set:
            classes_list.append(curr_class)
            classes_set.add(curr_class)
        row_list.append([annotations[i]["image"], classes_list.index(annotations[i][LABEL])])

with open(f"data_tools/ak_classification/dataset_train.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

row_list = [["image_directory", "label"]]

# do the same thing but for testing/validation data
f = open(f"{annotation_dir}{target_category}/test.json", "r")
annotations = json.load(f)
f.close()

for i in tqdm(range(len(annotations))):
    img = Image.open(f"{image_dir}{annotations[i]['image']}")
    width = img.width
    height = img.height
    if (height, width) in dimensions:
        dimensions[(height, width)] += 1
    else:
        dimensions[(height, width)] = 1
    if width == 640 and height == 360:
        curr_class = annotations[i][LABEL]
        if curr_class not in classes_set:
            classes_list.append(curr_class)
            classes_set.add(curr_class)
        row_list.append([annotations[i]["image"], classes_list.index(annotations[i][LABEL])])

with open(f"data_tools/ak_classification/dataset_test.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

# print out distribution of dimensions
print("Dimension Distribution:")
for key in dimensions:
    print(f"Dimensions count for {key}: {dimensions[key]}")