import math
import torch
import torch.nn as nn
from model.layers import Conv2D_BN
from utils.utils import getPadding

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding = 'same', groups = 1, dilation=1, bias = True):
        super(ResidualBlock, self).__init__()
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size = (1, 1), stride = stride[0], padding = 0)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size = kernel_size, stride = stride[1], padding = padding, groups = groups)
        self.conv2d_bn_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size = (1, 1), stride = stride[2], padding = 0)
        self.identity = Conv2D_BN(in_channels, activation, out_channels[2], kernel_size = (1, 1), stride = stride[1], padding = 0)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        output = self.conv2d_bn_3(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output += identity
        return output

class DenseBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding='same', groups=1, dilation=1, bias=True):
        super(DenseBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=stride[0], padding=0, groups=groups)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=stride[1], padding=padding)
        self.identity = Conv2D_BN(in_channels, activation, in_channels, kernel_size=(1, 1), stride=stride[1], padding=0)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output = torch.cat([output, identity], dim = 1)
        return output

class HourglassDownBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding='same', groups=1, dilation=1, bias=True):
        super(HourglassDownBlock, self).__init__()
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=stride[0], padding=0)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=stride[1], padding=padding, groups=groups)
        self.conv2d_bn_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size=(1, 1), stride=stride[2], padding=0)
        self.identity = Conv2D_BN(in_channels, activation, out_channels[2], kernel_size = (1, 1), stride = stride[1], padding = 0)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        output = self.conv2d_bn_3(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output += identity
        return output

class HourglassUpBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, padding=0, groups=1, dilation=1, mode = 'nearest', bias=True):
        super(HourglassUpBlock, self).__init__()
        if mode == 'nearest' or mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=1, padding=0)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=1, padding=padding, groups=groups)
        self.conv2d_bn_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size=(1, 1), stride=1, padding=0)
        self.identity = Conv2D_BN(in_channels, activation, out_channels[2], kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, input):
        input = self.up(input)
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        output = self.conv2d_bn_3(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output += identity
        return output

class DenseHourglassDownBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding='same', groups=1, dilation=1, bias=True):
        super(DenseHourglassDownBlock, self).__init__()
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=stride[0], padding=0)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=stride[1], padding=padding, groups=groups)
        self.identity = Conv2D_BN(in_channels, activation, in_channels, kernel_size = (1, 1), stride = stride[1], padding = 0)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output = torch.cat([output, identity], dim=1)
        return output

class DenseHourglassUpBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, padding=0, groups=1, dilation=1, mode = 'nearest', bias=True):
        super(DenseHourglassUpBlock, self).__init__()
        if mode == 'nearest' or mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=1, padding=0)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=1, padding=padding, groups=groups)
        self.identity = Conv2D_BN(in_channels, activation, in_channels, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, input):
        input = self.up(input)
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output = torch.cat([output, identity], dim=1)
        return output

class HourglassBlock(nn.Module):          #StackedHourGlass
    def __init__(self, activation, feature_num = 256, mode = "", bias = True):
        super(HourglassBlock, self).__init__()
        self.downblock1 = HourglassDownBlock(in_channels=int(feature_num / 4), activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), stride=(1, 2, 1), padding='same')
        self.downblock2 = HourglassDownBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), stride=(1, 2, 1), padding='same')

        self.sameblock = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')

        self.upblock1 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding='same', mode = mode)
        self.upblock2 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding='same', mode = mode)
        self.upblock3 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding='same', mode = mode)

        self.intermediateBlock1 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')
        self.intermediateBlock2 = ResidualBlock(in_channels=int(feature_num / 4), activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')
        self.intermediateBlock3 = ResidualBlock(in_channels=3, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), feature_num), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')

        self.output = Conv2D_BN(feature_num, activation=activation, out_channels=feature_num, kernel_size=(3, 3), stride = 1, padding='same')

    def forward(self, input):
        down_out1 = self.downblock1(input)
        down_out2 = self.downblock2(down_out1)

        same_out = self.sameblock(down_out2)

        intermediate_out1 = self.intermediateBlock1(down_out1)
        intermediate_out2 = self.intermediateBlock2(input)

        up_out1 = self.upblock1(same_out)
        up_out1 = up_out1 + intermediate_out1
        up_out2 = self.upblock2(up_out1)
        up_out2 = up_out2 + intermediate_out2
        heatmaps = self.upblock3(up_out2)

        output = self.output(heatmaps)
        return output

class DenseHourglassBlock(nn.Module):          #StackedHourGlass
    def __init__(self, activation, mode = "", bias = True):
        super(DenseHourglassBlock, self).__init__()
        self.downblock1 = DenseHourglassDownBlock(in_channels=64, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 2), padding='same')
        self.downblock2 = DenseHourglassDownBlock(in_channels=96, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 2), padding='same')

        self.sameblock = DenseBlock(in_channels=128, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same')

        self.upblock1 = DenseHourglassUpBlock(in_channels=160, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), padding='same', mode = mode)
        self.upblock2 = DenseHourglassUpBlock(in_channels=320, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), padding='same', mode = mode)
        self.upblock3 = DenseHourglassUpBlock(in_channels=448, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), padding='same', mode = mode)

        self.intermediateBlock1 = DenseBlock(in_channels=96, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.intermediateBlock2 = DenseBlock(in_channels=64, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same')

        self.output = Conv2D_BN(480, activation=activation, out_channels=512, kernel_size=(3, 3), stride = 1, padding='same')

    def forward(self, input):
        down_out1 = self.downblock1(input)
        down_out2 = self.downblock2(down_out1)

        same_out = self.sameblock(down_out2)
        intermediate_out1 = self.intermediateBlock1(down_out1)
        intermediate_out2 = self.intermediateBlock2(input)

        up_out1 = self.upblock1(same_out)
        up_out1 = torch.cat([up_out1, intermediate_out1], dim=1)
        up_out2 = self.upblock2(up_out1)
        up_out2 = torch.cat([up_out2, intermediate_out2], dim=1)
        heatmaps = self.upblock3(up_out2)

        output = self.output(heatmaps)
        return output