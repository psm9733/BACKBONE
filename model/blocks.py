import math
import torch
import torch.nn as nn
from model.layers import Conv2D_BN

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride = 1, padding = 0, groups = 1, dilation=1, bias = True):
        super(ResidualBlock, self).__init__()
        self.Conv2D_BN_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size = (1, 1), stride = stride, padding = 0)
        self.Conv2D_BN_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size = kernel_size, stride = 1, padding = padding, groups = groups)
        self.Conv2D_BN_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size = (1, 1), stride = 1, padding = 0)
        self.identity = Conv2D_BN(in_channels, activation, out_channels[2], kernel_size = (1, 1), stride = stride, padding = 0)

    def forward(self, input):
        output = self.Conv2D_BN_1(input)
        output = self.Conv2D_BN_2(output)
        output = self.Conv2D_BN_3(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output += identity
        return output

class DenseBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=True):
        super(DenseBlock, self).__init__()
        self.Conv2D_BN_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=stride,padding=0, groups=groups)
        self.Conv2D_BN_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=1, padding=padding)
        self.identity = Conv2D_BN(in_channels, activation, in_channels, kernel_size=(1, 1), stride=stride, padding=0)

    def forward(self, input):
        output = self.Conv2D_BN_1(input)
        output = self.Conv2D_BN_2(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output = torch.cat([output, identity], dim = 1)
        return output

class HourglassDownBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=True):
        super(HourglassDownBlock, self).__init__()
        self.Conv2D_BN_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=stride, padding=0)
        self.Conv2D_BN_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=1, padding=padding, groups=groups)
        self.Conv2D_BN_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size=(1, 1), stride=1, padding=0)
        self.identity = Conv2D_BN(in_channels, activation, out_channels[2], kernel_size = (1, 1), stride = stride, padding = 0)

    def forward(self, input):
        output = self.Conv2D_BN_1(input)
        output = self.Conv2D_BN_2(output)
        output = self.Conv2D_BN_3(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output += identity
        return output

class HourglassUpBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, padding=0, groups=1, dilation=1, bilinear = True, bias=True):
        super(HourglassUpBlock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.Conv2D_BN_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=1, padding=0)
        self.Conv2D_BN_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=1, padding=padding, groups=groups)
        self.Conv2D_BN_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size=(1, 1), stride=1, padding=0)
        self.identity = Conv2D_BN(in_channels, activation, out_channels[2], kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, input):
        input = self.up(input)
        output = self.Conv2D_BN_1(input)
        output = self.Conv2D_BN_2(output)
        output = self.Conv2D_BN_3(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output += identity
        return output