import matplotlib.pyplot as plt
import json
import os
import matplotlib.pyplot as plt

# draw a table of graphs. Each row will be from a file in data, and each column will be a different granularity
# the graphs will be the average accuracy of the model for each granularity

granularity_files = os.listdir("visualizer/data")

# example of data
# {
#     <model>: {
#         "<granularity>": {
#                 <data_split>: accuracy,
#                 ...
#             },
#         ...
#     }
# }

order_of_models = ["clip", "xclip"]
order_of_granularities = ["animal", "nothing"]
order_of_data_splits = ["ak_split", "head", "middle", "tail"]
data_split_lengths = {}

for i in order_of_data_splits:
    data_split_lengths[i] = 0

data = {}

for m in order_of_models:
    data[m] = {}
    for g in order_of_granularities:
        data[m][g] = {}
        for d in order_of_data_splits:
            data[m][g][d] = 0

for model in granularity_files:
    f = open("visualizer/data/" + model, "r")
    model_data = json.load(f)
    f.close()

    model = model[:model.find(".")]

    curr_acc = data[model]
    for key in model_data:
        cnt = 0
        total = 0
        pred = model_data[key]["cm_pred"]
        true = model_data[key]["cm_true"]
        for i in range(len(pred)):
            if pred[i] == true[i]:
                cnt += 1
            total += 1
        
        granularity = key[:key.find(':')]
        data_split = key[key.find(':')+1:]

        data_split_lengths[data_split] = total
        
        data[model][granularity][data_split] = cnt / total

fig, axs = plt.subplots(len(order_of_models), len(order_of_granularities), figsize=(len(order_of_models) * 5, len(order_of_granularities) * 5))

for i, model in enumerate(data):
    for j, granularity in enumerate(data[model]):
        axs[i][j].bar(data[model][granularity].keys(), data[model][granularity].values())

        axs[i][j].set_title(f"{model} - \"{granularity}\"")
        axs[i][j].set_ylim(0, 1.1)

        for k, v in enumerate(list(data[model][granularity].values())):
            axs[i][j].text(k, v, f"{(round(v, 2))} ({list(data_split_lengths.values())[k]})", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("visualizer/figures/overall_avgs.png")
plt.show()