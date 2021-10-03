import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN

class StemBlock(nn.Module):
    def __init__(self, in_channels, activation, bias=True):
        super(StemBlock, self).__init__()
        self.output_channels = 128
        self.output_stride = 4
        self.conv2d_1 = Conv2D_BN(in_channels, activation=activation, out_channels=32, kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_2 = Conv2D_BN(32, activation, out_channels=64, kernel_size=(3, 3), stride=2, padding='same', bias=bias)
        self.conv2d_3 = Conv2D_BN(64, activation, out_channels=32, kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_4 = Conv2D_BN(32, activation, out_channels=64, kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_5 = Conv2D_BN(64, activation, out_channels=self.output_channels, kernel_size=(3, 3), stride=2, padding='same', bias=bias)

    def forward(self, input):
        output1 = self.conv2d_1(input)
        output2 = self.conv2d_2(output1)
        output3 = self.conv2d_3(output2)
        output4 = self.conv2d_4(output3)
        shortcut = output2 + output4
        stemout = self.conv2d_5(shortcut)
        return stemout

    def getOutputChannels(self):
        return self.output_channels

    def getOutputStride(self):
        return self.output_stride

class SEBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_ratio, bias=True):
        super(SEBlock, self).__init__()
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