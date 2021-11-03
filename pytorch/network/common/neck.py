import logging
import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN

class FPN_3branch(nn.Module):
    def __init__(self, in_branch_channels, out_branch_channels, activation, bias=True):
        super().__init__()
        '''
        Args:
            in_branch_channels: [branch_1, branch_2, ... branch_N] branch 각각 연산 전 channels
            out_branch_channels: [branch_1, branch_2, ... branch_N] branch 각각 FPN 연산 후 channels
            activation: activation layer
            bias: con2d layer bias setting, default value is True
        '''
        if in_branch_channels != out_branch_channels:
            logging.debug("in_branch_channels and out_branch_channels are not the same.")

        self.in_branch_channels = in_branch_channels
        self.out_branch_channels = out_branch_channels

        self.conv2d_branch1 = nn.Sequential(
            Conv2D_BN(in_branch_channels[0], activation=activation, out_channels=out_branch_channels[0], kernel_size=(3, 3), stride=1, padding='same', bias=bias),
        )

        self.up_branch2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.identity_branch2 = Conv2D_BN(out_branch_channels[1], activation=activation, out_channels=out_branch_channels[0], kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_branch2 = nn.Sequential(
            Conv2D_BN(in_branch_channels[1], activation=activation, out_channels=out_branch_channels[1], kernel_size=(3, 3), stride=1, padding='same', bias=bias),
        )

        self.up_branch3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.identity_branch3 = Conv2D_BN(out_branch_channels[2], activation=activation, out_channels=out_branch_channels[1], kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_block_branch3 = nn.Sequential(
            Conv2D_BN(in_branch_channels[2], activation=activation, out_channels=out_branch_channels[2], kernel_size=(3, 3), stride=1, padding='same', bias=bias),
            Conv2D_BN(out_branch_channels[2], activation=activation, out_channels=int(out_branch_channels[2] / 2), kernel_size=(1, 1), stride=1, padding='same', bias=bias),
            Conv2D_BN(int(out_branch_channels[2] / 2), activation=activation, out_channels=out_branch_channels[2], kernel_size=(1, 1), stride=1, padding='same', bias=bias),
        )

    def forward(self, x1, x2, x3):
        branch3_out = self.conv2d_block_branch3(x3)
        branch3_up = self.up_branch3(branch3_out)

        if x2.shape[1] != branch3_up.shape[1]:
            branch3_up = self.identity_branch3(branch3_up)
        branch2_out = self.conv2d_branch2((x2 + branch3_up))
        branch2_up = self.up_branch2(branch2_out)

        if x1.shape[1] != branch2_up.shape[1]:
            branch2_up = self.identity_branch2(branch2_up)
        branch1_out = self.conv2d_branch1((x1 + branch2_up))
        return [branch1_out, branch2_out, branch3_out]

    def getOutputBranchChannels(self):
        return self.out_branch_channels

class PAN_3branch(nn.Module):               #todo
    def __init__(self, in_branch_channels, out_branch_channels, activation, bias=True):
        super().__init__()
        '''
        Args:
            in_branch_channels: [branch_1, branch_2, ... branch_N] branch 각각 연산 전 channels
            out_branch_channels: [branch_1, branch_2, ... branch_N] branch 각각 FPN 연산 후 channels
            activation: activation layer
            bias: con2d layer bias setting, default value is True
        '''
        if in_branch_channels != out_branch_channels:
            logging.debug("in_branch_channels and out_branch_channels are not the same.")
        self.in_branch_channels = in_branch_channels
        self.out_branch_channels = out_branch_channels

    def forward(self, x1, x2, x3):
        return [x1, x2, x3]