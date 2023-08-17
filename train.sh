#!/bin/bash

# python3 visualizer/calc_xclip.py -g nothing -p ak_split
# python3 visualizer/calc_xclip.py -g animal -p ak_split
# python3 visualizer/calc_xclip.py -g nothing -p head
# python3 visualizer/calc_clip.py -g nothing -p ak_split
# python3 visualizer/calc_clip.py -g animal -p ak_split

# have to create synonyms for the rest of the actions
# python3 visualizer/calc_xclip.py -g synonyms -p tail
# python3 visualizer/calc_clip.py -g synonyms -p tail
# python3 visualizer/calc_xclip.py -g synonyms -p middle
# python3 visualizer/calc_clip.py -g synonyms -p middle
# python3 visualizer/calc_xclip.py -g synonyms -p head
# python3 visualizer/calc_clip.py -g synonyms -p head

python3 visualizer/calc_clip.py -g animal_synonyms -p all

python3 visualizer/calc_xclip.py -g animal_synonyms -p ak_split
python3 visualizer/calc_xclip.py -g animal_synonyms -p tail
python3 visualizer/calc_xclip.py -g animal_synonyms -p middle
python3 visualizer/calc_xclip.py -g animal_synonyms -p head
python3 visualizer/calc_xclip.py -g animal_synonyms -p all

python3 visualizer/calc_clip.py -g nothing_synonyms -p ak_split
python3 visualizer/calc_clip.py -g nothing_synonyms -p tail
python3 visualizer/calc_clip.py -g nothing_synonyms -p middle
python3 visualizer/calc_clip.py -g nothing_synonyms -p head
python3 visualizer/calc_clip.py -g nothing_synonyms -p all

python3 visualizer/calc_xclip.py -g nothing_synonyms -p ak_split
python3 visualizer/calc_xclip.py -g nothing_synonyms -p tail
python3 visualizer/calc_xclip.py -g nothing_synonyms -p middle
python3 visualizer/calc_xclip.py -g nothing_synonyms -p head
python3 visualizer/calc_xclip.py -g nothing_synonyms -p all

python3 visualizer/calc_clip.py -g animal -p all
python3 visualizer/calc_clip.py -g nothing -p all
python3 visualizer/calc_xclip.py -g animal -p all
python3 visualizer/calc_xclip.py -g nothing -p all

# training scripts for budapest
# python3 main.py --wandb on --model berlin --epochs 100 --batch_size 64 --blocks 2 --num_classes 5
# python3 main.py --wandb on --model berlin --epochs 100 --batch_size 64 --blocks 4 --num_classes 5
# --------------------------------------------------------------------------------

# training scripts for beijing on AK data
# python3 main.py --wandb on --epochs 100 --batch_size 32 --blocks 2 --num_classes 5
# python3 main.py --wandb on --epochs 100 --batch_size 32 --blocks 3 --num_classes 5
# python3 main.py --wandb on --epochs 100 --batch_size 32 --blocks 4 --num_classes 5
# python3 main.py --wandb on --epochs 100 --batch_size 32 --blocks 5 --num_classes 5
# python3 main.py --wandb on --epochs 100 --batch_size 32 --blocks 3 --num_classes 5 --dropout 0.1
# python3 main.py --wandb on --epochs 100 --batch_size 32 --blocks 3 --num_classes 5 --dropout 0.2
# python3 main.py --wandb on --epochs 100 --batch_size 32 --blocks 3 --num_classes 5 --dropout 0.3
# python3 main.py --wandb on --epochs 100 --batch_size 32 --blocks 4 --num_classes 5 --activation relu
# python3 main.py --wandb on --epochs 100 --batch_size 32 --blocks 4 --num_classes 5 --activation tanh
# --------------------------------------------------------------------------------

# get blip2 and clip as a baseline
# extract the action from the blip2 output sentence
# add small multilayer NN to end of blip2/clip output as classifier
# fine tune blip2/clip on video grounding then do process above

# measure accuracy of blip2
# prompt is "x animal doing x action"
# test different granualirities (so change animal to species to parent species to eventually just "animal")
# measure using cosine function
# take top result and it's +1 if it's in one of the listed actions in AK

# get some more benchmarks w clip
# use clip for video based action recognition
# use a CNN on the clip output. CLIP produces x dimension vector. then take the outputs of the frames of the videos and stack them on top of each other. so x is the dimension of the clip embedding and y is the number of frames in the video

# overview of clip vs xclip prompting performance (compare static and dynamic)
# compare xclip results and paper results
# clip text adapter for ideas for prompting
# read xclip
# read https://arxiv.org/pdf/2110.04544.pdf