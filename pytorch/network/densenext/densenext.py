from network.common.layers import *
from network.densenext.blocks import *
import torch.nn as nn

class DenseNext18(nn.Module):
    def __init__(self, activation, in_channels, groups = 4, bias=True):
        super().__init__()
        self.output_stride = 16
        self.output_branch_channels = [in_channels + 32 * 2, in_channels + 32 * 4, in_channels + 32 * 6, in_channels + 32 * 8]
        self.output_channels = self.output_branch_channels[3]
        self.block1_1 = DenseNextBlock(in_channels=in_channels, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.block1_end = DenseNextBlock(in_channels=in_channels + 32, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)

        self.block2_t = TransitionLayer(in_channels=in_channels + 32 * 2, activation=activation, out_channels=in_channels + 32 * 2, kernel_size=(1, 1), stride=2, padding='same')
        self.block2_1 = DenseNextBlock(in_channels=in_channels + 32 * 2, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.block2_end = DenseNextBlock(in_channels=in_channels + 32 * 3, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)

        self.block3_t = TransitionLayer(in_channels=in_channels + 32 * 4, activation=activation, out_channels=in_channels + 32 * 4, kernel_size=(1, 1), stride=2, padding='same')
        self.block3_1 = DenseNextBlock(in_channels=in_channels + 32 * 4, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.block3_end = DenseNextBlock(in_channels=in_channels + 32 * 5, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)

        self.block4_t = TransitionLayer(in_channels=in_channels + 32 * 6, activation=activation, out_channels=in_channels + 32 * 6, kernel_size=(1, 1), stride=2, padding='same')
        self.block4_1 = DenseNextBlock(in_channels=in_channels + 32 * 6, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)
        self.block4_end = DenseNextBlock(in_channels=in_channels + 32 * 7, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=(1, 1, 1), padding='same', groups = groups, bias=bias)

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

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride