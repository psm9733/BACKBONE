from network.common.layers import *
from network.resnext.blocks import *
import torch.nn as nn

class ResNext14(nn.Module):
    def __init__(self, activation, in_channels, groups=32, bias=True):
        super(ResNext14, self).__init__()
        self.block1_end = ResidualBlock(in_channels=64, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups)
        self.block2_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups)
        self.block3_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups)
        self.output = ResidualBlock(in_channels=1024, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups)
        self.output_channel = 2048
        self.resnext14 = nn.Sequential(
            Conv2D_BN(in_channels, activation=activation, out_channels=64, kernel_size=(7, 7), stride=2, padding=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),
            self.block1_end,
            self.block2_end,
            self.block3_end,
            self.output
        )

    def forward(self, input):
        output = self.resnext14(input)
        return output

    def getOutputChannel(self):
        return self.output_channel

class ResNext26(nn.Module):
    def __init__(self, activation, in_channels, groups=32, bias=True):
        super(ResNext26, self).__init__()
        self.block1_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups)
        self.block2_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups)
        self.block3_end = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups)
        self.output = ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups)
        self.resnext26 = nn.Sequential(
            Conv2D_BN(in_channels, activation=activation, out_channels=64, kernel_size=(7, 7), stride=2, padding='same'),
            nn.MaxPool2d((2, 2), stride=2),
            ResidualBlock(in_channels=64, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups),
            self.block1_end,

            ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups),
            self.block2_end,

            ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups),
            self.block3_end,

            ResidualBlock(in_channels=1024, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups),
            self.output
        )
        self.output_channel = 2048

    def forward(self, input):
        output = self.resnext26(input)
        return output

    def getOutputChannel(self):
        return self.output_channel

class ResNext50(nn.Module):
    def __init__(self, activation, in_channels, groups = 32, bias = True):
        super(ResNext50, self).__init__()
        self.block1_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block2_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block3_end = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.output = ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)

        self.resnext50 = nn.Sequential(
            Conv2D_BN(in_channels, activation=activation, out_channels=64, kernel_size=(7, 7), stride = 2, padding=(3, 3), bias=bias),
            nn.MaxPool2d((2, 2), stride=2),

            ResidualBlock(in_channels=64, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias),
            ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias),
            self.block1_end,

            ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups, bias=bias),
            ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias),
            ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias),
            self.block2_end,

            ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups, bias=bias),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias),
            self.block3_end,

            ResidualBlock(in_channels=1024, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups, bias=bias),
            ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias),
            self.output
        )
        self.output_channel = 2048

    def forward(self, input):
        output = self.resnext50(input)
        return output

    def getOutputChannel(self):
        return self.output_channel