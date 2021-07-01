import math
import torch.nn as nn

class Conv2D_BN(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'):
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
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'):
        super(ResidualBlock, self).__init__()
        self.Conv2D_BN_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size = (1, 1), stride = (1, 1), padding = 0)
        self.Conv2D_BN_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size = kernel_size, stride = stride, padding = padding)
        self.Conv2D_BN_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size = (1, 1), stride = (1, 1), padding = 0)
        self.identity = Conv2D_BN(in_channels, activation, out_channels[2], kernel_size = (1, 1), stride = (1, 1), padding = 0)

    def forward(self, input):
        output = self.Conv2D_BN_1(input)
        output = self.Conv2D_BN_2(output)
        output = self.Conv2D_BN_3(output)
        if output.shape[1] != input.shape[1]:
            identity = self.identity(input)
        else:
            identity = input
        output += identity
        return output