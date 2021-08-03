from network.common.layers import *
from network.densenext.blocks import *
import torch.nn as nn

class DenseNext32(nn.Module):
    def __init__(self, activation, groups = 32, bias=True):
        super(DenseNext32, self).__init__()
        self.block1_end = DenseBlock(in_channels=128, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.block2_end = DenseBlock(in_channels=224, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.block3_end = DenseBlock(in_channels=352, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.output = DenseBlock(in_channels=512, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.densenext32 = nn.Sequential(
            Conv2D_BN(1, activation=activation, out_channels=64, kernel_size=(7, 7), stride=2, padding='same', bias=bias),
            nn.MaxPool2d((2, 2), stride=2),

            DenseBlock(in_channels=64, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=96, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            self.block1_end,

            TransitionLayer(in_channels=160, activation=activation, out_channels=160, kernel_size=(1, 1), stride=2, padding='same'),
            DenseBlock(in_channels=160, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=192, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            self.block2_end,

            TransitionLayer(in_channels=256, activation=activation, out_channels=256, kernel_size=(1, 1), stride=2, padding='same'),
            DenseBlock(in_channels=256, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=288, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=320, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            self.block3_end,

            TransitionLayer(in_channels=384, activation=activation, out_channels=384, kernel_size=(1, 1), stride=2, padding='same'),
            DenseBlock(in_channels=384, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=416, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=448, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=480, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias),
            self.output
        )
        self.output_channel = 544

    def forward(self, input):
        output = self.densenext32(input)
        return output

    def getOutputChannel(self):
        return self.output_channel

class DenseNext64(nn.Module):
    def __init__(self, activation, groups = 32, bias=True):
        super(DenseNext64, self).__init__()
        self.block1_end = DenseBlock(in_channels=192, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias)
        self.block2_end = DenseBlock(in_channels=384, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias)
        self.block3_end = DenseBlock(in_channels=672, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias)
        self.output = DenseBlock(in_channels=1024, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias)
        self.densenext64 = nn.Sequential(
            Conv2D_BN(3, activation=activation, out_channels=64, kernel_size=(7, 7), stride=2, padding='same', bias=bias),
            nn.MaxPool2d((2, 2), stride=2),

            DenseBlock(in_channels=64, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=96, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=128, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=160, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            self.block1_end,

            TransitionLayer(in_channels=224, activation=activation, out_channels=224, kernel_size=(1, 1), stride=2, padding='same', bias=bias),
            DenseBlock(in_channels=224, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=256, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=288, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=320, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=352, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            self.block2_end,

            TransitionLayer(in_channels=416, activation=activation, out_channels=416, kernel_size=(1, 1), stride=2, padding='same', bias=bias),
            DenseBlock(in_channels=416, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=448, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=480, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=512, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=544, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=576, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=608, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=640, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            self.block3_end,

            TransitionLayer(in_channels=704, activation=activation, out_channels=704, kernel_size=(1, 1), stride=2, padding='same', bias=bias),
            DenseBlock(in_channels=704, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=736, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=768, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=800, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=832, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=864, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=896, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=928, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=960, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            DenseBlock(in_channels=992, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1), padding='same', groups = groups, bias=bias),
            self.output
        )
        self.output_channel = 1056

    def forward(self, input):
        output = self.densenext64(input)
        return output

    def getOutputChannel(self):
        return self.output_channel