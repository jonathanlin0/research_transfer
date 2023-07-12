# this file creates the csv file that is used in ak_dataloader
# it creates a CSV file where the first row is the path to the image and the second row is the label

import csv
import json
from tqdm import tqdm

LABEL = "animal_parent_class"

annotation_dir = "datasets/Animal_Kingdom/pose_estimation/annotation/"

target_category = "ak_P1"

row_list = [["image_directory", "label"]]

# do this for training data
f = open(f"{annotation_dir}{target_category}/train.json", "r")
annotations = json.load(f)
f.close()

for i in tqdm(range(len(annotations))):
    row_list.append([annotations[i]["image"], annotations[i][LABEL]])

with open(f"data_tools/ak_classification_data_train.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

# do the same thing but for testing/validation data
f = open(f"{annotation_dir}{target_category}/test.json", "r")
annotations = json.load(f)
f.close()

for i in tqdm(range(len(annotations))):
    row_list.append([annotations[i]["image"], annotations[i][LABEL]])

with open(f"data_tools/ak_classification_data_test.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(row_list)