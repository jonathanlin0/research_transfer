import csv
import json
import pandas as pd

PE_DATA_PATH_TRAIN = "datasets/Animal_Kingdom/pose_estimation/annotation/ak_P1/train.json"
PE_DATA_PATH_VAL = "datasets/Animal_Kingdom/pose_estimation/annotation/ak_P1/test.json"

AR_DATA_PATH_TRAIN = "datasets/Animal_Kingdom/action_recognition/annotation/train.csv"
AR_DATA_PATH_VAL = "datasets/Animal_Kingdom/action_recognition/annotation/test.csv"

DF_ACTION_PATH = "datasets/Animal_Kingdom/action_recognition/annotation/df_action.xlsx"
DATA_OUTPUT_PATH = "visualizer/"

action_list = set()
action_key = pd.read_excel(DF_ACTION_PATH)
data = {}

# Load the Excel file into a DataFrame
excel_file_path = "datasets/Animal_Kingdom/action_recognition/AR_metadata.xlsx"
df = pd.read_excel(excel_file_path)

data = {}

# Iterate through rows
for index, row in df.iterrows():
    # Access columns using row[column_name] or row[column_index]
    label = row["labels"]
    if "," not in label:
        data[row["video_id"]] = {
            "animal": row["list_animal"].replace("[", "").replace("]", "").replace("'", "").split(", ")[0].lower(),
            "animal_parent_class": row["list_animal_parent_class"].replace("[", "").replace("]", "").replace("'", "").split(", ")[0].lower(),
            "action": action_key.at[int(label), 'action'].lower()
        }
        action_list.add(action_key.at[int(label), 'action'].lower())

# get all actions


data_copy = data.copy()
data = {}
data["action_index_key"] = list(action_list)
data["video_data"] = data_copy

# write the json file data
f = open(DATA_OUTPUT_PATH + "data.json", "w")
json.dump(data, f, indent=4)
f.close()