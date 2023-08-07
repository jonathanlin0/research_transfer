import csv
import json

annotation_dir = "/Users/jonathanlin/Documents/GitHub/research_transfer/datasets/Animal_Kingdom/action_recognition/annotation"

types = ["train", "val"]

for t in types:
    with open(f"{annotation_dir}/{t}.csv", 'r') as csvfile:

        # Step 2: Create a CSV reader object
        csvreader = csv.reader(csvfile, delimiter=' ')
        
        data = {}
        
        # Step 3: Iterate through the rows
        for i, row in enumerate(csvreader):
            # Step 4: Process each row
            if i == 0:
                continue
            video_name = row[0]
            labels = row[4]
            data[video_name] = labels
        
        data = list(data.items())
        for i in range(len(data)):
            data[i] = list(data[i])
        
        with open(f"data_tools/ak_ar_videos/{t}.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(data)