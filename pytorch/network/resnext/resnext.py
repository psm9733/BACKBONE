from network.common.layers import *
from network.resnext.blocks import *
import torch.nn as nn

class ResNext12(nn.Module):
    def __init__(self, activation, in_channels, groups=32, bias=True):
        super(ResNext12, self).__init__()
        self.output_channel = 512
        self.block1_end = ResidualBlock(in_channels=in_channels, activation=activation, out_channels=(32, 32, 64), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block2_end = ResidualBlock(in_channels=64, activation=activation, out_channels=(64, 64, 128), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block3_end = ResidualBlock(in_channels=128, activation=activation, out_channels=(128, 128, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block4_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(256, 256, self.output_channel), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)

    def forward(self, input):
        block1_out = self.block1_end(input)
        block2_out = self.block2_end(block1_out)
        block3_out = self.block3_end(block2_out)
        block4_out = self.block4_end(block3_out)
        return [block1_out, block2_out, block3_out, block4_out]

    def getOutputChannel(self):
        return self.output_channel

class ResNext24(nn.Module):
    def __init__(self, activation, in_channels, groups=32, bias=True):
        super(ResNext24, self).__init__()
        self.output_channel = 2048
        self.block1_1 = ResidualBlock(in_channels=in_channels, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block1_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block2_1 = ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups, bias=bias)
        self.block2_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block3_1 = ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups, bias=bias)
        self.block3_end = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block4_1 = ResidualBlock(in_channels=1024, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups, bias=bias)
        self.block4_end = ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)

    def forward(self, input):
        block1_1_out = self.block1_1(input)
        block1_out = self.block1_end(block1_1_out)
        block2_1_out = self.block2_1(block1_out)
        block2_out = self.block2_end(block2_1_out)
        block3_1_out = self.block3_1(block2_out)
        block3_out = self.block3_end(block3_1_out)
        block4_1_out = self.block4_1(block3_out)
        block4_out = self.block4_end(block4_1_out)
        return [block1_out, block2_out, block3_out, block4_out]

    def getOutputChannel(self):
        return self.output_channel

class ResNext48(nn.Module):
    def __init__(self, activation, in_channels, groups = 32, bias = True):
        super(ResNext48, self).__init__()
        self.output_channel = 2048
        self.block1_1 = ResidualBlock(in_channels=in_channels, activation=activation, out_channels=(64, 64, 256) ,kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block1_2 = ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block1_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)

        self.block2_1 = ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups, bias=bias)
        self.block2_2 = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block2_3 = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block2_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)

        self.block3_1 = ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups, bias=bias)
        self.block3_2 = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block3_3 = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block3_4 = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block3_5 = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block3_end = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)

        self.block4_1 = ResidualBlock(in_channels=1024, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 2, 1), padding='same', groups=groups, bias=bias)
        self.block4_2 = ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)
        self.block4_end = ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups=groups, bias=bias)

    def forward(self, input):
        block1_1_out = self.block1_1(input)
        block1_2_out = self.block1_2(block1_1_out)
        block1_out = self.block1_end(block1_2_out)

        block2_1_out = self.block2_1(block1_out)
        block2_2_out = self.block2_2(block2_1_out)
        block2_3_out = self.block2_3(block2_2_out)
        block2_out = self.block2_end(block2_3_out)

        block3_1_out = self.block3_1(block2_out)
        block3_2_out = self.block3_2(block3_1_out)
        block3_3_out = self.block3_3(block3_2_out)
        block3_4_out = self.block3_4(block3_3_out)
        block3_5_out = self.block3_4(block3_4_out)
        block3_out = self.block3_end(block3_5_out)

        block4_1_out = self.block4_1(block3_out)
        block4_2_out = self.block4_2(block4_1_out)
        block4_out = self.block4_end(block4_2_out)
        return [block1_out, block2_out, block3_out, block4_out]

    def getOutputChannel(self):
        return self.output_channel