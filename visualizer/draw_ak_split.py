import matplotlib.pyplot as plt
import json

AVERAGE_BAR_COLOR = "green"
ANIMAL_CLASS_BAR_COLOR = "blue"

# get appropriate data for the animal kingdom dataset split
f = open("visualizer/data/xclip.json", "r")
data = json.load(f)
f.close()

# remove irrelevant info
for key in list(data.keys()):
    if "ak_split" in key:
        data[key[:key.find(":")]] = data[key]
    del data[key]

num_classes = 5

fig, axs = plt.subplots(len(data), num_classes + 1, figsize=((num_classes + 1) * 7, len(data) * 5))

for i, key in enumerate(data):
    # curr_data holds the accuracy for each animal type
    curr_data = {}
    overall_avg = []
    for animal_data in data[key]["raw_data"].values():

        animal_type = animal_data["animal"]
        if animal_data["pred_action"] == animal_data["true_action"]:
            curr_data[animal_type] = curr_data.get(animal_type, []) + [1]
            overall_avg.append(1)
        else:
            curr_data[animal_type] = curr_data.get(animal_type, []) + [0]
            overall_avg.append(0)

    # sort the data by the key so the values are in alphabetical order
    curr_data = dict(sorted(curr_data.items()))

    # add in the average
    curr_data_list = list(curr_data.items())
    curr_data_list.insert(0, ("AVERAGE", overall_avg))

    curr_data = dict(curr_data_list)

    # for item in list(curr_data.keys()):
    #     curr_data[item] = curr_data[item]
    #     del curr_data[item]

    lengths = []
    for item in curr_data:
        lengths.append(len(curr_data[item]))
        curr_data[item] = sum(curr_data[item]) / len(curr_data[item])

    color = [AVERAGE_BAR_COLOR] + [ANIMAL_CLASS_BAR_COLOR] * (len(curr_data) - 1)

    # plot the data
    axs[i][0].bar(list(curr_data.keys()), list(curr_data.values()), color=color, alpha=0.7)
    axs[i][0].set_title(f'Granularity = "{key}"')

    for j, v in enumerate(list(curr_data.values())):
        axs[i][0].text(j, v, f"{(round(v, 2))} ({lengths[j]})", ha='center', va='bottom', fontsize=9)
    
    axs[i][0].set_ylim(0, 1.1)
    axs[i][0].tick_params(axis='x', labelsize=10)

    # now graph the data for each animal class
    curr_data = dict(list(curr_data.items())[1:])
    for key in list(curr_data.keys()):
        pass

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()