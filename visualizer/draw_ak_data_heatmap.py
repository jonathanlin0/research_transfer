import matplotlib.pyplot as plt
import numpy as np
import copy
import json
from matplotlib.colors import LinearSegmentedColormap

actions = {
    "moving": 0,
    "eating": 0,
    "attending": 0,
    "swimming": 0,
    "sensing": 0,
    "keeping still": 0,
}

animal_classes = ["insect", "lizard", "primate", "spider", "water bird"]

data = {}

for animal in animal_classes:
    data[animal] = copy.deepcopy(actions)

data_file_path = "visualizer/data/clip.json"

f = open(data_file_path, "r")
model_data = json.load(f)
f.close()

for key in model_data:
    if "ak_split" in key:
        for file_path in model_data[key]["raw_data"]:
            # if model_data[key]["raw_data"][file_path]["animal"] in animal_classes:
            action = model_data[key]["raw_data"][file_path]["true_action"]
            animal = model_data[key]["raw_data"][file_path]["animal"]
            data[animal][action] += 1
        break


# Extract actions and classes
actions = list(data[list(data.keys())[0]].keys())
classes = list(data.keys())

# Convert the data into a 2D array for the heatmap
heatmap_data = np.array([[data[class_name][action] for action in actions] for class_name in classes])

# Create a colormap from black to dark blue
cmap = plt.cm.Blues

# Set 0 values to black
cmap.set_under("black")

# Create the heatmap
plt.figure(figsize=(10, 6))  # Set the figure size

heatmap = plt.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=0.1, vmax=60)

# Set x and y ticks
plt.xticks(np.arange(len(actions)), actions, rotation=45, ha="right", fontsize=14)
plt.yticks(np.arange(len(classes)), classes, fontsize=14)

# Set colorbar
plt.colorbar(heatmap)

# Add numbers in the center of each rectangle
for i in range(len(classes)):
    for j in range(len(actions)):
        value = heatmap_data[i, j]
        if value != 0:
            plt.text(j, i, str(value), ha="center", va="center", color="black", fontsize=16)

plt.xlabel("Actions", fontsize=16)
plt.ylabel("Classes", fontsize=16)
plt.title("Heatmap of Data Points by Class and Action", fontsize=18)

plt.tight_layout()
plt.savefig("visualizer/figures/ak_data_heatmap.png")
plt.show()