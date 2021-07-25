import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN
import torch.nn as nn

class StemBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding = 'same', groups = 1, dilation=1, bias = True):
        super(StemBlock, self).__init__()
        self.out_channels = out_channels
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation=activation, out_channels=out_channels[0], kernel_size=kernel_size, stride=stride[0], padding=padding)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation=activation, out_channels=out_channels[1], kernel_size=kernel_size, stride=stride[1], padding=padding)
        if len(out_channels) == 3:
            self.conv2d_bn_3 = Conv2D_BN(out_channels[1], activation=activation, out_channels=out_channels[2], kernel_size=kernel_size, stride=stride[2], padding=padding)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        if len(self.out_channels) == 3:
            output = self.conv2d_bn_3(output)
        return output