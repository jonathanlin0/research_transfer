import torch
from torch import nn

from typing import Type

class Block_Beijing(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels

        super(Block_Beijing, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm_1 = nn.BatchNorm2d(self.out_channels)
        self.activation_1 = activation
        
        self.conv_2 = nn.Conv2d(
            in_channels = self.out_channels,
            out_channels = self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm_2 = nn.BatchNorm2d(self.out_channels)
        self.activation_2 = activation

        # downsample adjusts the original input to fit the shape
        self.downsample = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = 1,
            stride = 2,
            padding = 0,
            bias=False
        )

    def forward(self, x):
        identity = x

        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.norm_2(x)

        # apply identity function
        x = x + self.downsample(identity)
        x = self.activation_2(x)

        return x  


class beijing(nn.Module):

    def __init__(
            self,
            img_channels: int,
            block: Type[Block_Beijing],
            num_blocks: int,
            activation,
            num_classes: int):
        super().__init__()


        self.prep = nn.Conv2d(
                in_channels = img_channels,
                out_channels = 64,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                bias = False
        )
        self.norm_1 = nn.BatchNorm2d(64)
        self.activation = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        layers = []
        current_in = 64
        # create number of blocks according to the num_blocks input
        for i in range(1, num_blocks + 1):
            layers.append(block(current_in, current_in * 2, activation = activation))
            current_in *= 2
        
        self.temp = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * (2 ** num_blocks), num_classes)
        

    def forward(self, x):
        
        x = self.prep(x)
        x = self.norm_1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.temp(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x