import matplotlib.pyplot as plt
import json
import os
import copy

AVERAGE_OVERALL_BAR_COLOR = "red"
AVERAGE_CLASS_BAR_COLOR = "green"
AVERAGE_ACTION_BAR_COLOR = "blue"

granularity_files = os.listdir("visualizer/data")
order_of_models = ["clip", "xclip"]
granularities = ["animal", "nothing"]
animals = ["insect", "lizard", "primate", "spider", "water bird"]
valid_actions = ["moving", "eating", "attending", "swimming", "sensing", "keeping still"]

fig, axs = plt.subplots(len(order_of_models) * 2, len(animals) + 1, figsize=((len(animals) + 1) * 4, len(order_of_models) * 2 * 3))

# example of data
# {
#     <model> :{
#         <granularity>: {
#             <class>: {
#                 <action>: <accuracy>,
#                 ...
#             },
#             ...
#         }, ...
#     },
#     ...
# }
data = {}

for m in order_of_models:
    data[m] = {}

    for g in granularities:
        data[m][g] = {}

        for a in animals:
            data[m][g][a] = {}

            for action in valid_actions:
                data[m][g][a][action] = []

# example of averages
# {
#     <model>: {
#         <granularity>: {
#             <class>: accuarcy,
#             ...
#         },
#         ...
#     }
#     ...
# }
averages = {}

for m in order_of_models:
    averages[m] = {}

    for g in granularities:
        averages[m][g] = {}

        for a in animals:
            averages[m][g][a] = []

# example of model_avgs
# {
#     <model>: {
#         <granularity>: accuracy,
#         ...
#     },
#     ...
# }
model_avgs = {}

for m in order_of_models:
    model_avgs[m] = {}

    for g in granularities:
        model_avgs[m][g] = []


for model in granularity_files:
    f = open("visualizer/data/" + model, "r")
    model_data = json.load(f)
    f.close()

    model = model[:model.find(".")]

    for i, key in enumerate(model_data):
        # curr_data holds the accuracy for each animal type
        granularity = key[:key.find(':')]
        data_split = key[key.find(':')+1:]

        if data_split == "ak_split":
            overall_avg = []

            for animal_data in model_data[key]["raw_data"].values():

                animal_type = animal_data["animal"]
                if animal_data["pred_action"] == animal_data["true_action"]:
                    data[model][granularity][animal_type][animal_data["true_action"]] += [1]
                    averages[model][granularity][animal_type] += [1]
                    overall_avg.append(1)
                else:
                    data[model][granularity][animal_type][animal_data["true_action"]] += [0]
                    averages[model][granularity][animal_type] += [0]
                    overall_avg.append(0)
    
            model_avgs[model][granularity] = overall_avg
                        

        # # sort the data by the key so the values are in alphabetical order
        # curr_data = dict(sorted(curr_data.items()))

        # # add in the average
        # curr_data_list = list(curr_data.items())
        # curr_data_list.insert(0, ("AVERAGE", overall_avg))

        # curr_data = dict(curr_data_list)

        # lengths = []
        # for item in curr_data:
        #     lengths.append(len(curr_data[item]))
        #     curr_data[item] = sum(curr_data[item]) / len(curr_data[item])

        # color = [AVERAGE_BAR_COLOR] + [ANIMAL_CLASS_BAR_COLOR] * (len(curr_data) - 1)

        # # plot the data
        # axs[i][0].bar(list(curr_data.keys()), list(curr_data.values()), color=color, alpha=0.7)
        # axs[i][0].set_title(f'Granularity = "{key}"')

        # for j, v in enumerate(list(curr_data.values())):
        #     axs[i][0].text(j, v, f"{(round(v, 2))} ({lengths[j]})", ha='center', va='bottom', fontsize=9)
        
        # axs[i][0].set_ylim(0, 1.1)
        # axs[i][0].tick_params(axis='x', labelsize=10)    
    
data_lengths = copy.deepcopy(data)
averages_lengths = copy.deepcopy(averages)
model_avgs_lengths = copy.deepcopy(model_avgs)

# calc the averages
for i, model in enumerate(data):
    for j, granularity in enumerate(data[model]):
        for k, animal in enumerate(data[model][granularity]):
            for l, action in enumerate(data[model][granularity][animal]):
                data_lengths[model][granularity][animal][action] = len(data_lengths[model][granularity][animal][action])
                if len(data[model][granularity][animal][action]) == 0:
                    data[model][granularity][animal][action] = None
                else:
                    data[model][granularity][animal][action] = sum(data[model][granularity][animal][action]) / len(data[model][granularity][animal][action])
            averages_lengths[model][granularity][animal] = len(averages_lengths[model][granularity][animal])
            if len(averages[model][granularity][animal]) == 0:
                averages[model][granularity][animal] = None
            else:
                averages[model][granularity][animal] = sum(averages[model][granularity][animal]) / len(averages[model][granularity][animal])
        model_avgs_lengths[model][granularity] = len(model_avgs_lengths[model][granularity])
        if len(model_avgs[model][granularity]) == 0:
            model_avgs[model][granularity] = None
        else:
            model_avgs[model][granularity] = sum(model_avgs[model][granularity]) / len(model_avgs[model][granularity])

# plot the data
for i, model in enumerate(data):
    row = i + order_of_models.index(model)
    for j, granularity in enumerate(data[model]):
        row_offset = granularities.index(granularity)

        # graph overall averages
        x_values = ["AVERAGE"] + list(averages[model][granularity].keys())
        y_values = [model_avgs[model][granularity]] + list(averages[model][granularity].values())

        colors = [AVERAGE_OVERALL_BAR_COLOR] + [AVERAGE_CLASS_BAR_COLOR] * (len(y_values) - 1)

        lengths = [model_avgs_lengths[model][granularity]] + list(averages_lengths[model][granularity].values())
        for j, v in enumerate(y_values):
            axs[row + row_offset][0].text(j, v, f"{(round(v, 2))} ({lengths[j]})", ha='center', va='bottom', fontsize=8)
        axs[row + row_offset][0].bar(x_values, y_values, alpha=0.7, color=colors)
        axs[row + row_offset][0].set_ylim(0, 1.1)
        axs[row + row_offset][0].tick_params(axis='x', labelsize=9)
        axs[row + row_offset][0].set_title(f'Model = {model}; Granularity = "{granularity}"')

        # graph data for each animal class
        for k, animal in enumerate(data[model][granularity]):
            x_values = ["AVERAGE"] + list(data[model][granularity][animal].keys())
            y_values = [averages[model][granularity][animal]] + list(data[model][granularity][animal].values())

            # modify data for the cases where there is no data point of an action for a given class
            for l, action in enumerate(y_values):
                if action == None:
                    y_values[l] = 0.00
                    x_values[l] = "N/A"

            colors = [AVERAGE_CLASS_BAR_COLOR] + [AVERAGE_ACTION_BAR_COLOR] * (len(y_values) - 1)
            lengths = [averages_lengths[model][granularity][animal]] + list(data_lengths[model][granularity][animal].values())
            bars = axs[row + row_offset][k + 1].bar(x_values, y_values, alpha=0.7, color = colors)

            # Add labels on top of each bar
            # have to do this weird method for some reason cuz the values weren't being centered w the bars
            for bar, length in zip(bars, lengths):
                height = bar.get_height()
                axs[row + row_offset][k + 1].text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,  # Adjust this value for vertical position of the label
                    f"{(round(height, 2))} ({length})",
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
            
            axs[row + row_offset][k + 1].set_ylim(0, 1.1)
            axs[row + row_offset][k + 1].tick_params(axis='x', labelsize=7)
            axs[row + row_offset][k + 1].set_title(f'Animal = "{animal}"')

plt.tight_layout()
plt.savefig("visualizer/figures/ak_split.png")
plt.show()