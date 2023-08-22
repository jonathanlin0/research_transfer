import pandas as pd
from transformers import AutoProcessor, AutoModel
from contextlib import redirect_stdout, redirect_stderr
import json
import argparse
import av
import copy
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import os

df = pd.read_excel("/Users/jonathanlin/Documents/GitHub/research_transfer/datasets/Animal_Kingdom/action_recognition/AR_metadata.xlsx")

root_path = "/Users/jonathanlin/Documents/GitHub/research_transfer/datasets/Animal_Kingdom/action_recognition/dataset/video/"

data = []

for i in range(len(df)):
    vid_path = root_path + df.at[i, "video_id"] + ".mp4"
    container = av.open(vid_path)
    data.append([container.streams.video[0].frames, vid_path])

print(sorted(data, reverse=True)[:5])
    