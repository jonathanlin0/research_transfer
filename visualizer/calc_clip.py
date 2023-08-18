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
from PIL import Image
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '-g', '--granularity', default='class',
    type = str,
    required = True,
    help='set the granularity of the CLIP model',
    choices=["class", "animal", "nothing", "animal_synonyms", "nothing_synonyms"]
)
parser.add_argument(
    '-p', '--data_portion', default='all',
    type = str,
    required = False,
    help='set how the text labels are put into CLIP',
    choices=["all", "head", "middle", "tail", "ak_split"]
)
args = vars(parser.parse_args())
granularity = args["granularity"]
data_portion = args["data_portion"]
print(f"[INFO]: Set the model to {os.path.basename(__file__)[os.path.basename(__file__).find('calc_') + 5:os.path.basename(__file__).find('.')]}")
print(f"[INFO]: Set the granularity to {granularity}")
print(f"[INFO]: Set the data portion to {data_portion}")

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

def calculate(vid_path, text_labels):
    # with redirect_stdout(None), redirect_stderr(None):
    if True:
        container = av.open(vid_path)

        values = []
        i = 0

        frames = []

        # add every 8 frames
        while i + 7 < container.streams.video[0].frames:
            cap = cv2.VideoCapture(vid_path)

            # Check if the video file was successfully opened
            if not cap.isOpened():
                print("Error: Could not open video file.")
                exit()
            
            # Get the desired frame index
            frame_index = i  # Replace this with the index of the frame you want to extract

            # Set the video capture object to the desired frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            # Read the frame
            ret, frame = cap.read()
            frames.append(frame)

            i += 8
        
        frames = np.array(frames)

        inputs = processor(
            text=list(text_labels.keys()),
            images=frames,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits_per_image = outputs.logits_per_image  # this is the video-text similarity score

        for subarray in logits_per_image:
            subarray.softmax(dim=0)
            values.append(subarray.cpu())
        
        arr = np.array([0] * len(list(text_labels.keys())))
        # get average of list of numpy arrays
        for array in values:
            arr = np.add(arr, array)
        
        arr = arr.squeeze().tolist()

    for i in range(len(arr)):
        arr[i] = arr[i] / len(values)
    
    # find the average value of the probabilities in each section
    actions = {}

    for pair in text_labels:
        actions[text_labels[pair]] = 0
    
    for i in range(len(arr)):
        actions[text_labels[list(text_labels.keys())[i]]] += arr[i]
    
    dict_copy = dict((sorted(list(actions.items()))))

    # sort dict_copy by the value (value is the average probability of the action from CLIP)
    dict_copy = dict(sorted(dict_copy.items(), key=lambda item: item[1], reverse=True))

    return list(dict_copy.items())[0][0]


f = open("visualizer/data.json", "r")
data = json.load(f)
f.close()


df = pd.read_excel("datasets/Animal_Kingdom/action_recognition/annotation/df_action.xlsx")
# modify data according to data_portion
if data_portion != "all":
    if data_portion == "head" or data_portion == "middle" or data_portion == "tail":
        correct_actions = []
        for index, row in df.iterrows():
            if row["segment"] == data_portion:
                correct_actions.append(row["action"].lower())
        
        # remove actions from action["action_index_key"]
        to_remove = []
        for action in correct_actions:
            if action not in data["action_index_key"]:
                to_remove.append(action)
        for action in to_remove:
            correct_actions.remove(action)
        data["action_index_key"] = correct_actions

        # remove data from data["video_data"]
        data_temp = {}
        for vid_path in data["video_data"]:
            if data["video_data"][vid_path]["action"] in correct_actions:
                data_temp[vid_path] = data["video_data"][vid_path]
        
        data["video_data"] = data_temp
    elif data_portion == "ak_split":
        # this is the split used in animal kingdom
        # the 5 animals are lizards, primates, spiders, orthopteran insects, water fowl
        # the namings arent exactly the same, because an animal has multiple types of names
        valid_actions = ["moving", "eating", "attending", "swimming", "sensing", "keeping still"]
        classes = ["lizard", "primate", "spider", "insect", "water bird"]
        modified_to_original = {}

        subclass_df = pd.read_excel("datasets/Animal_Kingdom/action_recognition/AR_metadata.xlsx", sheet_name="Animal")
        for index, row in subclass_df.iterrows():
            curr_classes = row["Sub-Class"].split(" / ")
            # turn all to lowercase
            for i in range(len(curr_classes)):
                curr_classes[i] = curr_classes[i].lower()

            matching_class = ""
            for curr_class in curr_classes:
                if curr_class in classes:
                    matching_class = curr_class
                    break
            
            if matching_class != "":
                for curr_class in curr_classes:
                    modified_to_original[curr_class] = matching_class
        
        # modify data obj to only have these 5 classes with the valid actions
        i = 0
        while i < len(data["action_index_key"]):
            if data["action_index_key"][i] not in valid_actions:
                data["action_index_key"].pop(i)
            else:
                i += 1
        
        data_copy = {}
        for data_pt in data["video_data"]:
            if data["video_data"][data_pt]["animal"].lower() in modified_to_original and data["video_data"][data_pt]["action"] in valid_actions:
                data_copy[data_pt] = data["video_data"][data_pt]
                data_copy[data_pt]["animal"] = modified_to_original[data["video_data"][data_pt]["animal"].lower()]
        
        data["video_data"] = data_copy

synonyms_dict = json.load(open("visualizer/synonyms.json", "r"))

# get all subclasses
classes = set()
for video in data["video_data"]:
    classes.add(data["video_data"][video]["animal_parent_class"].lower())

def get_strings_class():
    out = {}
    for action in data["action_index_key"]:
        for class_ in classes:
            out[f"a {class_} is {action}"] = action
    return out

def get_strings_animal():
    out = {}
    for action in data["action_index_key"]:
        out[f"an animal is {action}"] = action
    return out

def get_strings_nothing():
    out = {}
    for action in data["action_index_key"]:
        out[f"{action}"] = action
    return out

def get_strings_animal_synonyms():
    out = {}
    for action in data["action_index_key"]:
        out[f"an animal is {action}"] = action
        for synonym in synonyms_dict[action]:
            out[f"an animal is {synonym}"] = action

    return out

def get_strings_nothing_synonyms():
    out = {}
    for action in data["action_index_key"]:
        out[f"{action}"] = action
        for synonym in synonyms_dict[action]:
            out[f"{synonym}"] = action

    return out

granularity = args["granularity"]

# create training strings
strings = {}
if granularity == "class":
    strings = get_strings_class()
elif granularity == "animal":
    strings = get_strings_animal()
elif granularity == "nothing":
    strings = get_strings_nothing()
elif granularity == "animal_synonyms":
    strings = get_strings_animal_synonyms()
elif granularity == "nothing_synonyms":
    strings = get_strings_nothing_synonyms()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

video_root = "datasets/Animal_Kingdom/action_recognition/dataset/video/"

# cm stands for confusion matrix
cm_pred = []
cm_true = []

output_data = {
    "cm_pred": cm_pred,
    "cm_true": cm_true,
    "raw_data": {}
}

video_data = data["video_data"]
# remove all but the first 5 pairings of video_data
for vid_path, _ in tqdm(video_data.items()):
    correct_label = video_data[vid_path]["action"]

    full_vid_path = video_root + vid_path + ".mp4"

    with redirect_stdout(None), redirect_stderr(None):
        num_frames = av.open(full_vid_path).streams.video[0].frames

    # only doing videos w 8 or more frames. have to fix later
    if num_frames >= 8:
        pred_label = calculate(full_vid_path, strings)

        cm_pred.append(data["action_index_key"].index(pred_label))
        cm_true.append(data["action_index_key"].index(correct_label))

        output_data["raw_data"][vid_path] = {
            "animal_parent_class": video_data[vid_path]["animal_parent_class"],
            "animal": video_data[vid_path]["animal"],
            "pred_action": pred_label,
            "true_action": correct_label
        }


json_data = {}
try:
    f = open(f"visualizer/data/clip.json", "r")
    json_data = json.load(f)
    f.close()
except:
    pass
json_data[granularity + ":" + data_portion] = output_data
f = open(f"visualizer/data/clip.json", "w")
json.dump(json_data, f, indent=4)
f.close()