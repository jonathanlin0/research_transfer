from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

cwd = "/home/jonathan/Desktop/Perona_Research"

def create_csv():
    # categories = ["ak_P1", "ak_P2", "ak_P3_amphibian", "ak_P3_bird", "ak_P3_bird", "ak_P3_fish", "ak_P3_mammal", "ak_P3_reptile"]
    categories = ["ak_P1", "ak_P2"]
    data_types = ["test", "train"] # the dataset is split into test.json and train.json
    out_csv = "image_directory"
    
    # add headers to beginning
    temp_file = open(cwd + "/datasets/Animal_Kingdom/pose_estimation/annotation/ak_P1/test.json", "r")
    temp_data = json.load(temp_file)
    for i in range(len(temp_data[0]["joints"])):
        out_csv += (",part" + str(i) + "_x,part" + str(i) + "_y")
    out_csv += "\n"

    for category in categories:
        for data_type in data_types:
            f = open(cwd + "/datasets/Animal_Kingdom/pose_estimation/annotation/" + category + "/" + data_type + ".json", "r")
            data = json.load(f)
            
            for item in data:
                out_csv += (item["image"] + ",")
                joint_locations = item["joints"]
                for joint in joint_locations:
                    out_csv += (str(joint[0]) + "," + str(joint[1]) + ",")
                out_csv = out_csv[:-1]
                out_csv += "\n"
            f.close()
    
    f = open(cwd + "/data_tools/PE_data.csv", "w")
    f.write(out_csv)
    f.close()


create_csv()