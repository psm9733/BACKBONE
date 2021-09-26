import torch.nn as nn
from network.common.layers import Conv2D_BN
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding = 'same', groups = 1, dilation=1, bias = True):
        super(ResidualBlock, self).__init__()
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size = (1, 1), stride = stride[0], padding = 0, bias=bias)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size = kernel_size, stride = stride[1], padding = padding, groups = groups, bias=bias)
        self.conv2d_bn_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size = (1, 1), stride = stride[2], padding = 0, bias=bias)
        self.identity = Conv2D_BN(in_channels, activation, out_channels[2], kernel_size = (1, 1), stride = stride[1], padding = 0, bias=bias)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        output = self.conv2d_bn_3(output)
        if output.shape[1:4] == input.shape[1:4]:
            identity = input
        else:
            identity = self.identity(input)
        output += identity
        return output

class HourglassDownBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding='same', groups=1, dilation=1, bias=True):
        super(HourglassDownBlock, self).__init__()
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=stride[0], padding=0, groups=groups, bias=bias)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=stride[1], padding=padding, groups=groups, bias=bias)
        self.conv2d_bn_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size=(1, 1), stride=stride[2], padding=0, groups=groups, bias=bias)
        self.identity = Conv2D_BN(in_channels, activation, out_channels[2], kernel_size = (1, 1), stride = stride[1], padding = 0, bias=bias)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        output = self.conv2d_bn_3(output)
        if output.shape[1:4] == input.shape[1:4]:
            identity = input
        else:
            identity = self.identity(input)
        output += identity
        return output

class HourglassUpBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, padding=0, groups=1, dilation=1, mode = 'nearest', bias=True):
        super(HourglassUpBlock, self).__init__()
        if mode == 'nearest' or mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=bias)
        self.conv2d_bn_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size=(1, 1), stride=1, padding=0, bias=bias)
        self.conv2d_bn_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=bias)
        self.conv2d_bn_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size=(1, 1), stride=1, padding=0, bias=bias)

    def forward(self, input1, input2):
        output = input1 + input2
        output = self.up(output)
        output = self.conv2d_bn_1(output)
        output = self.conv2d_bn_2(output)
        output = self.conv2d_bn_3(output)
        return output

class HourglassBlock(nn.Module):          #StackedHourGlass
    def __init__(self, activation, in_feature, feature_num = 256, mode = "", padding = 'same', groups = 32, bias = True):
        super(HourglassBlock, self).__init__()
        self.downblock1 = HourglassDownBlock(in_channels=in_feature, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(4, 4), stride=(1, 2, 1), padding=padding, groups = 1, bias=bias)
        self.downblock2 = HourglassDownBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(4, 4), stride=(1, 2, 1), padding=padding, groups = groups, bias=bias)
        self.downblock3 = HourglassDownBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(4, 4), stride=(1, 2, 1), padding=padding, groups = groups, bias=bias)

        self.sameblock1 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding=padding, groups = groups, bias=bias)
        self.sameblock2 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding=padding, groups = groups, bias=bias)

        self.upblock1 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding=padding, groups = groups, mode = mode, bias=bias)
        self.upblock2 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding=padding, groups = groups, mode = mode, bias=bias)
        self.upblock3 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding=padding, groups = groups, mode = mode, bias=bias)

        self.intermediateBlock1 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding=padding, groups = groups, bias=bias)
        self.intermediateBlock2 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding=padding, groups = groups, bias=bias)
        self.intermediateBlock3 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding=padding, groups = groups, bias=bias)

    def forward(self, input):
        down_out1 = self.downblock1(input)
        down_out2 = self.downblock2(down_out1)
        down_out3 = self.downblock2(down_out2)

        same_out1 = self.sameblock1(down_out3)
        same_out2 = self.sameblock2(same_out1)

        intermediate_out1 = self.intermediateBlock1(down_out1)
        intermediate_out2 = self.intermediateBlock2(down_out2)
        intermediate_out3 = self.intermediateBlock3(down_out3)

        up_out1 = self.upblock1(intermediate_out3, same_out2)
        up_out2 = self.upblock2(intermediate_out2, up_out1)
        heatmap = self.upblock3(intermediate_out1, up_out2)
        return heatmap