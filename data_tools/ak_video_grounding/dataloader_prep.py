import csv
import json
from tqdm import tqdm

VIDEO_DIR = "datasets/Animal_Kingdom/video_grounding/annotation/ak_vg_duration.json"

# training data
f = open("datasets/Animal_Kingdom/video_grounding/annotation/train.txt", "r")
video_labels = f.read().splitlines()
f.close()

# sanity check to ensure data is in right format
for i in tqdm(range(len(video_labels))):
    row = video_labels[i].split(" ")
    assert len(row) >= 3
    assert "##" in row[2]

row_list = [["video_directory", "start_time", "end_time", "caption"]]
for i in tqdm(range(len(video_labels))):
    row = video_labels[i].split(" ")
    row = row[:2] + row[2].split("##") + row[3:]
    row = row[:3] + [" ".join(row[3:])]
    row[0] += ".mp4"
    row_list.append(row)

with open(f"data_tools/ak_video_grounding/dataset_train.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(row_list)


# testing/validation data
f = open("datasets/Animal_Kingdom/video_grounding/annotation/test.txt", "r")
video_labels = f.read().splitlines()
f.close()

# sanity check to ensure data is in right format
for i in tqdm(range(len(video_labels))):
    row = video_labels[i].split(" ")
    assert len(row) >= 3
    assert "##" in row[2]

row_list = [["video_directory", "start_time", "end_time", "caption"]]
for i in tqdm(range(len(video_labels))):
    row = video_labels[i].split(" ")
    row = row[:2] + row[2].split("##") + row[3:]
    row = row[:3] + [" ".join(row[3:])]
    row[0] += ".mp4"
    row_list.append(row)

with open(f"data_tools/ak_video_grounding/dataset_test.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(row_list)