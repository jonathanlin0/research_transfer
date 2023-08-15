import csv
import json
import pandas as pd

PE_DATA_PATH_TRAIN = "datasets/Animal_Kingdom/pose_estimation/annotation/ak_P1/train.json"
PE_DATA_PATH_VAL = "datasets/Animal_Kingdom/pose_estimation/annotation/ak_P1/test.json"

AR_DATA_PATH_TRAIN = "datasets/Animal_Kingdom/action_recognition/annotation/train.csv"
AR_DATA_PATH_VAL = "datasets/Animal_Kingdom/action_recognition/annotation/test.csv"

DF_ACTION_PATH = "datasets/Animal_Kingdom/action_recognition/annotation/df_action.xlsx"
DATA_OUTPUT_PATH = "visualizer/"

# read in PE data
f = open(PE_DATA_PATH_TRAIN, "r")
pe_data_raw = json.load(f)
f.close()
f = open(PE_DATA_PATH_VAL, "r")
pe_data_raw += json.load(f)
f.close()

all_image_paths = set()
for image in pe_data_raw:
    all_image_paths.add(image["image"][:image["image"].find("/")])

# convert PE data to a dictionary
pe_data = {}
for image in pe_data_raw:
    pe_data[image["image"][:image["image"].find("/")]] = image

action_list = set()
df = pd.read_excel(DF_ACTION_PATH)
data = {}

# iterate through AR data and add the data that has corresponding PE data (for the class, parent class, etc data)
f = open(AR_DATA_PATH_TRAIN, "r")
reader = csv.reader(f, delimiter=" ")
for row in reader:
    file_path = row[0]
    actions = row[4]
    if file_path in all_image_paths and "," not in actions:
        data[file_path] = {
            "animal": pe_data[file_path]["animal"],
            "animal_parent_class": pe_data[file_path]["animal_parent_class"],
            "animal_class": pe_data[file_path]["animal_class"],
            "animal_subclass": pe_data[file_path]["animal_subclass"],
            "action": actions
        }
        action_list.add(df.at[int(actions), 'action'].lower())

data_copy = data.copy()
data = {}
data["action_index_key"] = list(action_list)
data["video_data"] = data_copy


# write the json file data
f = open(DATA_OUTPUT_PATH + "data.json", "w")
json.dump(data, f, indent=4)
f.close()