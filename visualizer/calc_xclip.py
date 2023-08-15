from transformers import AutoProcessor, AutoModel
from contextlib import redirect_stdout, redirect_stderr
import json
import argparse
from transformers import XCLIPVisionModel, XCLIPVisionConfig
import av
import copy
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '-g', '--granularity', default='class',
    type = str,
    required = True,
    help='set the granularity of the XCLIP model',
    choices=["class", "animal", "nothing"]
)
parser.add_argument(
    '-p', '--data_portion', default='all',
    type = str,
    required = True,
    help='set how the text labels are put into XCLIP',
    choices=["all", "head", "middle", "tail"]
)
args = vars(parser.parse_args())
granularity = args["granularity"]
print(f"[INFO]: Set the granularity to {granularity}")

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
    with redirect_stdout(None), redirect_stderr(None):
        container = av.open(vid_path)
        values = []
        i = 0


        # num of frames is 8
        while i + 7 < container.streams.video[0].frames:
            video = read_video_pyav(container, np.arange(i, i + 8))
            inputs = processor(
                text=list(text_labels.keys()),
                videos=list(video),
                return_tensors="pt",
                padding=True,
            ).to(device)

            # forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
            probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
            # index = torch.argmax(probs)

            values.append(probs.cpu())
            i += 8 # (overlap 2 frames)
        
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

    # sort dict_copy by the value (value is the average probability of the action from XCLIP)
    dict_copy = dict(sorted(dict_copy.items(), key=lambda item: item[1], reverse=True))

    return list(dict_copy.items())[0][0]

# get all subclasses
classes = set()
f = open("visualizer/data.json", "r")
data = json.load(f)
f.close()

for video in data["video_data"]:
    classes.add(data["video_data"][video]["animal_class"].lower())


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

granularity = args["granularity"]

# create training strings
strings = {}
if granularity == "class":
    strings = get_strings_class()
elif granularity == "animal":
    strings = get_strings_animal()
elif granularity == "nothing":
    strings = get_strings_nothing()


processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32").to(device)

df = pd.read_excel("datasets/Animal_Kingdom/action_recognition/annotation/df_action.xlsx")

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
min_val = 9999
# remove all but the first 5 pairings of video_data
for vid_path, _ in tqdm(video_data.items()):
    label_num = video_data[vid_path]["action"]

    full_vid_path = video_root + vid_path + ".mp4"

    with redirect_stdout(None), redirect_stderr(None):
        num_frames = av.open(full_vid_path).streams.video[0].frames

    # only doing videos w 8 or more frames. have to fix later
    if num_frames >= 8:
        correct_label = df.at[int(label_num), 'action'].lower()

        pred_label = calculate(full_vid_path, strings)

        cm_pred.append(data["action_index_key"].index(pred_label))
        cm_true.append(data["action_index_key"].index(correct_label))

        output_data["raw_data"][vid_path] = {
            "animal": video_data[vid_path]["animal"],
            "animal_class": video_data[vid_path]["animal_class"],
            "animal_subclass": video_data[vid_path]["animal_subclass"],
            "pred_action": pred_label,
            "true_action": correct_label
        }
    
json_data = {}
try:
    f = open(f"visualizer/data/xclip.json", "r")
    json_data = json.load(f)
    f.close()
except:
    pass
json_data[granularity] = output_data
f = open(f"visualizer/data/xclip.json", "w")
json.dump(json_data, f, indent=4)
f.close()