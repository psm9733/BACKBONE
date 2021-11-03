from network.common.layers import Conv2D_BN
from network.common.blocks import SEBlock
import torch.nn as nn

class XBlock(nn.Module):            #based ResidualBlock + group convolution
    def __init__(self, activation, in_channels, block_width, bottleneck_ratio, stride, padding='same', groups=1, dilation=1, bias=True):
        super().__init__()
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, int(block_width / bottleneck_ratio), kernel_size=(1, 1), stride=1, padding=padding, dilation=dilation, bias=bias)
        self.conv2d_bn_2 = Conv2D_BN(int(block_width / bottleneck_ratio), activation, int(block_width / bottleneck_ratio), kernel_size=(3, 3), stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias)
        self.conv2d_bn_3 = Conv2D_BN(int(block_width / bottleneck_ratio), activation, block_width, kernel_size=(1, 1), stride=1, padding=padding, dilation=dilation, bias=bias)
        self.identity = Conv2D_BN(in_channels, activation, block_width, kernel_size=(1, 1), stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        output = self.conv2d_bn_3(output)
        if output.shape[2:4] == input.shape[2:4]:
            identity = input
        else:
            identity = self.identity(input)
        output += identity
        return output

class YBlock(nn.Module): # based ResidualBlock + group convolution + SEBlock
    def __init__(self, activation, in_channels, block_width, bottleneck_ratio, stride, padding='same', groups=1, dilation=1, bias=True):
        super().__init__()
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, int(block_width / bottleneck_ratio), kernel_size=(1, 1), stride=1, padding=padding, dilation=dilation, bias=bias)
        self.conv2d_bn_2 = Conv2D_BN(int(block_width / bottleneck_ratio), activation, int(block_width / bottleneck_ratio), kernel_size=(3, 3), stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias)
        self.seblock = SEBlock(int(block_width / bottleneck_ratio), bottleneck_ratio=4, bias=bias)
        self.conv2d_bn_3 = Conv2D_BN(int(block_width / bottleneck_ratio), activation, block_width, kernel_size=(1, 1), stride=1, padding=padding, dilation=dilation, bias=bias)
        self.identity = Conv2D_BN(in_channels, activation, block_width, kernel_size=(1, 1), stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        output = self.seblock(output)
        output = self.conv2d_bn_3(output)
        if output.shape[2:4] == input.shape[2:4]:
            identity = input
        else:
            identity = self.identity(input)
        output += identity
        return output