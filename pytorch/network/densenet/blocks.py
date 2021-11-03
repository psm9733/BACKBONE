import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN

class DenseBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding='same', groups=1, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=stride[0], padding=0, groups=groups)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=stride[1], padding=padding)
        self.identity = Conv2D_BN(in_channels, activation, in_channels, kernel_size=(1, 1), stride=stride[1], padding=0)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        if output.shape[1:4] == input.shape[1:4]:
            identity = input
        else:
            identity = self.identity(input)
        output = torch.cat([output, identity], dim = 1)
        return output