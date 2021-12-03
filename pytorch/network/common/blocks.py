import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN

class StemBlock_3(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, activation, bias=True):
        super().__init__()
        self.output_stride = 4
        self.out_channels_list = out_channels_list
        self.conv2d_1 = Conv2D_BN(in_channels_list[0], activation=activation, out_channels=out_channels_list[0], kernel_size=(3, 3), stride=2, padding='same', bias=bias)
        self.conv2d_2 = Conv2D_BN(in_channels_list[1], activation, out_channels=out_channels_list[1], kernel_size=(3, 3), stride=2, padding='same', bias=bias)
        self.conv2d_3 = Conv2D_BN(in_channels_list[2], activation, out_channels=out_channels_list[2], kernel_size=(3, 3), stride=1, padding='same', bias=bias)

    def forward(self, input):
        output1 = self.conv2d_1(input)
        output2 = self.conv2d_2(output1)
        stemout = self.conv2d_3(output2)
        return stemout

    def getOutputChannels(self):
        return self.out_channels_list[len(self.out_channels_list) - 1]

    def getOutputStride(self):
        return self.output_stride

class StemBlock_5(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, activation, bias=True):
        super().__init__()
        self.output_stride = 4
        self.out_channels_list = out_channels_list
        self.conv2d_1 = Conv2D_BN(in_channels_list[0], activation=activation, out_channels=out_channels_list[0], kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_2 = Conv2D_BN(in_channels_list[1], activation, out_channels=out_channels_list[1], kernel_size=(3, 3), stride=2, padding='same', bias=bias)
        self.conv2d_3 = Conv2D_BN(in_channels_list[2], activation, out_channels=out_channels_list[2], kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_4 = Conv2D_BN(in_channels_list[3], activation, out_channels=out_channels_list[3], kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_5 = Conv2D_BN(in_channels_list[4], activation, out_channels=out_channels_list[4], kernel_size=(3, 3), stride=2, padding='same', bias=bias)

    def forward(self, input):
        output1 = self.conv2d_1(input)
        output2 = self.conv2d_2(output1)
        output3 = self.conv2d_3(output2)
        output4 = self.conv2d_4(output3)
        shortcut = output2 + output4
        stemout = self.conv2d_5(shortcut)
        return stemout

    def getOutputChannels(self):
        return self.out_channels_list[len(self.out_channels_list) - 1]

    def getOutputStride(self):
        return self.output_stride

class SEBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_ratio, bias=True):
        super().__init__()
        self.out_channels = in_channels
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, int(self.out_channels / bottleneck_ratio), (1, 1), bias=bias),
            nn.ReLU(),
            nn.Conv2d(int(self.out_channels / bottleneck_ratio), self.out_channels, (1, 1), bias=bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        batch, channel, height, width = input.size()
        output = self.squeeze(input)
        output = output.view(batch, channel, 1, 1)
        output = self.excitation(output)
        output = output.view(batch, channel, 1, 1)
        return input * output

    def getOutputChannels(self):
        return self.output_channels

