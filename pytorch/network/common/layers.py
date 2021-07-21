import math
import torch.nn as nn
from utils.utils import getPadding

class Conv2D_BN(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, mode = "bn", stride = 1, padding = 'same', groups = 1, dilation=1, bias = True):
        super(Conv2D_BN, self).__init__()
        self.padding = getPadding(kernel_size, padding)
        self.activation = activation
        self.mode = mode
        in_channels = math.floor(in_channels)
        out_channels = math.floor(out_channels)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = self.padding, dilation = dilation, groups = groups, bias = bias)
        if mode == "bn":
            self.batchNorm_layer = nn.BatchNorm2d(out_channels)
        elif mode == "in":
            self.instanceNorm_layer = nn.InstanceNorm2d(out_channels)

    def forward(self, input):
        output = self.conv_layer(input)
        if self.mode == "bn":
            output = self.batchNorm_layer(output)
        elif self.mode == "in":
            output = self.instanceNorm_layer(output)
        if self.activation != None:
            output = self.activation(output)
        return output

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=True, padding_mode='zeros'):
        super(TransitionLayer, self).__init__()
        self.padding = getPadding(kernel_size, padding)
        self.conv2d_BN = Conv2D_BN(in_channels, activation, out_channels, kernel_size=kernel_size, stride = stride, padding=self.padding)
        # self.Max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride = 2)

    def forward(self, input):
        output = self.conv2d_BN(input)
        # output = self.Max_pool(output)
        return output