import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN

class StemBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding = 'same', groups = 1, dilation=1, bias = True):
        super(StemBlock, self).__init__()
        self.out_channels = out_channels
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation=activation, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride[0], padding=padding, bias=bias)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation=activation, out_channels=out_channels[1], kernel_size=kernel_size, stride=stride[1], padding=padding, bias=bias)
        if len(out_channels) == 3:
            self.conv2d_bn_3 = Conv2D_BN(out_channels[1], activation=activation, out_channels=out_channels[2], kernel_size=kernel_size, stride=stride[2], padding=padding, bias=bias)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        if len(self.out_channels) == 3:
            output = self.conv2d_bn_3(output)
        return output

class DenseBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding='same', groups=1, dilation=1, bias=True):
        super(DenseBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=stride[0], padding=0, groups=groups, bias=bias)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=stride[1], padding=padding, bias=bias)
        self.identity = Conv2D_BN(in_channels, activation, in_channels, kernel_size=(1, 1), stride=stride[1], padding=0, bias=bias)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output = torch.cat([output, identity], dim = 1)
        return output