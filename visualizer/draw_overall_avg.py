import matplotlib.pyplot as plt
import json
import os

# draw a table of graphs. Each row will be from a file in data, and each column will be a different granularity
# the graphs will be the average accuracy of the model for each granularity

granularity_files = os.listdir("visualizer/data")

# example of data
# {
#     <model>: {
#         "<granularity>": <accuracy>,
#         ...
#     }
# }
data = {}

for file in granularity_files:
    f = open("visualizer/data/" + file, "r")
    model_data = json.load(f)
    f.close()

    curr_acc = {}
    for granularity in model_data:
        cnt = 0
        total = 0
        pred = model_data[granularity]["cm_pred"]
        true = model_data[granularity]["cm_true"]
        for i in range(len(pred)):
            if pred[i] == true[i]:
                cnt += 1
            total += 1
        curr_acc[granularity] = cnt / total
    data[file[:file.find(".")]] = curr_acc

print(data)