#!/bin/bash

# training scripts for budapest
python3 main.py --wandb on --model budapest --epochs 100 --batch_size 64 --num_classes 5
python3 main.py --wandb on --model berlin --epochs 100 --batch_size 64 --blocks 2 --num_classes 5
python3 main.py --wandb on --model berlin --epochs 100 --batch_size 64 --blocks 4 --num_classes 5
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