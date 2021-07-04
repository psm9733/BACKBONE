import math
import torch
import torch.nn as nn

class Conv2D_BN(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride = 1, padding = 0, groups = 1, dilation=1, bias = True, padding_mode = 'zeros'):
        super(Conv2D_BN, self).__init__()
        in_channels = math.floor(in_channels)
        out_channels = math.floor(out_channels)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias, padding_mode = padding_mode)
        self.batchNorm_layer = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, input):
        output = self.conv_layer(input)
        output = self.batchNorm_layer(output)
        output = self.activation(output)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride = 1, padding = 0, groups = 1, dilation=1, bias = True, padding_mode = 'zeros'):
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
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=True, padding_mode='zeros'):
        super(DenseBlock, self).__init__()
        self.out_channels_num = len(out_channels)
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

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=True, padding_mode='zeros'):
        super(TransitionLayer, self).__init__()
        self.Conv2D_BN = Conv2D_BN(in_channels, activation, out_channels, kernel_size=kernel_size, stride = stride, padding=0)
        # self.Max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride = 2)

    def forward(self, input):
        output = self.Conv2D_BN(input)
        # output = self.Max_pool(output)
        return output