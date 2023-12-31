import pandas as pd
import csv
from tqdm import tqdm
import json

SEGMENTS = ["head", "middle"]
MAX_FRAMES_FROM_VID = 25

annotation_dir = "datasets/Animal_Kingdom/action_recognition/annotation"
img_dir = "datasets/Animal_Kingdom/action_recognition/dataset/image"

valid_action_indexes = []

df = pd.read_excel("/Users/jonathanlin/Documents/GitHub/research_transfer/datasets/Animal_Kingdom/action_recognition/annotation/df_action.xlsx")

for i in range(len(df)):
    if df.at[i, "segment"] in SEGMENTS:
        valid_action_indexes.append(str(i))

last_vid = ""

# add the conversion from converted action index to original index
converted = {}
for i in range(len(valid_action_indexes)):
    converted[int(i)] = int(valid_action_indexes[i])
# write converted to a json file
with open("data_tools/ak_ar_images/converted.json", "w") as file:
    json.dump(converted, file)

# do this for training data
row_list = [["image_directory", "label"]]
landmarks_frame = pd.read_csv(f"{annotation_dir}/train.csv", delimiter = " ")
cnt = 0
for i in tqdm(range(len(landmarks_frame))):
    actions = landmarks_frame.iloc[i, 4].split(",")
    converted_actions = []
    # check if actions are in the valid data segments (head, middle, tail)
    valid = True
    for action in actions:
        if action not in valid_action_indexes:
            valid = False
            break
        converted_actions.append(str(valid_action_indexes.index(action)))
    # only add a max of n amount of frames from the same video
    if landmarks_frame.iloc[i, 0] == last_vid:
        cnt += 1
    else:
        cnt = 1
    if cnt >= MAX_FRAMES_FROM_VID:
        valid = False
    last_vid = landmarks_frame.iloc[i, 0]
    if valid:
        row_list.append([f"{landmarks_frame.iloc[i, 3]}", ",".join(converted_actions)])

with open(f"data_tools/ak_ar_images/train.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

# do this for validation data
row_list = [["image_directory", "label"]]
landmarks_frame = pd.read_csv(f"{annotation_dir}/val.csv", delimiter = " ")
cnt = 0
for i in tqdm(range(len(landmarks_frame))):
    actions = landmarks_frame.iloc[i, 4].split(",")
    converted_actions = []
    # check if actions are in the valid data segments (head, middle, tail)
    valid = True
    for action in actions:
        if action not in valid_action_indexes:
            valid = False
            break
        converted_actions.append(str(valid_action_indexes.index(action)))
    # only add a max of n amount of frames from the same video
    if landmarks_frame.iloc[i, 0] == last_vid:
        cnt += 1
    else:
        cnt = 1
    if cnt >= MAX_FRAMES_FROM_VID:
        valid = False
    last_vid = landmarks_frame.iloc[i, 0]
    if valid:
        row_list.append([f"{landmarks_frame.iloc[i, 3]}", ",".join(converted_actions)])

with open(f"data_tools/ak_ar_images/val.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(row_list)