import json

f = open("/Users/jonathanlin/Downloads/data/clip.json", "r")
model_data = json.load(f)
f.close()

f = open("visualizer/data/clip.json", "r")
og_data = json.load(f)
f.close()

print(len(og_data))

og_data.update(model_data)

f = open("visualizer/data/clip.json", "w")
json.dump(og_data, f, indent = 4)
f.close()

print(len(og_data))