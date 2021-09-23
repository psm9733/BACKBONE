from network.common.layers import *
from network.densenet.blocks import *
import torch.nn as nn

class DenseNet18(nn.Module):
    def __init__(self, activation, in_channels, groups = 1, bias=True):
        super(DenseNet18, self).__init__()
        self.output_channel = in_channels + 32 * 8
        self.block1_1 = DenseBlock(in_channels=in_channels, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.block1_end = DenseBlock(in_channels=in_channels + 32, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)

        self.block2_t = TransitionLayer(in_channels=in_channels + 32 * 2, activation=activation, out_channels=in_channels + 32 * 2, kernel_size=(1, 1), stride=2, padding='same')
        self.block2_1 = DenseBlock(in_channels=in_channels + 32 * 2, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.block2_end = DenseBlock(in_channels=in_channels + 32 * 3, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)

        self.block3_t = TransitionLayer(in_channels=in_channels + 32 * 4, activation=activation, out_channels=in_channels + 32 * 4, kernel_size=(1, 1), stride=2, padding='same')
        self.block3_1 = DenseBlock(in_channels=in_channels + 32 * 4, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.block3_end = DenseBlock(in_channels=in_channels + 32 * 5, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)

        self.block4_t = TransitionLayer(in_channels=in_channels + 32 * 6, activation=activation, out_channels=in_channels + 32 * 6, kernel_size=(1, 1), stride=2, padding='same')
        self.block4_1 = DenseBlock(in_channels=in_channels + 32 * 6, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.block4_end = DenseBlock(in_channels=in_channels + 32 * 7, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)


    def forward(self, input):
        block1_1_out = self.block1_1(input)
        block1_out = self.block1_end(block1_1_out)

        block2_t_out = self.block2_t(block1_out)
        block2_1_out = self.block2_1(block2_t_out)
        block2_out = self.block2_end(block2_1_out)

        block3_t_out = self.block3_t(block2_out)
        block3_1_out = self.block3_1(block3_t_out)
        block3_out = self.block3_end(block3_1_out)

        block4_t_out = self.block4_t(block3_out)
        block4_1_out = self.block4_1(block4_t_out)
        block4_out = self.block4_end(block4_1_out)

        return [block1_out, block2_out, block3_out, block4_out]

    def getOutputChannel(self):
        return self.output_channel