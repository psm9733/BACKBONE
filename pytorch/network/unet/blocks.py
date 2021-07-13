import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN
import torch.nn.functional as F

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

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNetUpBlock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = UNetSameBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = UNetSameBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetSameBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(UNetSameBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # Conv2D_BN(in_channels, activation=nn.ReLU(), out_channels=mid_channels, kernel_size=3, stride=1, padding=0),
            # Conv2D_BN(mid_channels, activation=nn.ReLU(), out_channels=out_channels, kernel_size=3, stride=1, padding=0)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(UNetDownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UNetSameBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UNetOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetOutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sig(x)
        return x