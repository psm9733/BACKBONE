import torch
from network.common.layers import *
from network.shnet.blocks import *
from network.common.modules import SAM
import torch.nn as nn

class SHNet(nn.Module):          #StackedHourGlass
    def __init__(self, activation, in_channels, feature_num = 512, groups = 32, mode = "", bias = True):
        super(SHNet, self).__init__()
        self.block1_end = HourglassBlock(activation, in_channels, feature_num, mode, padding='same', groups = groups)
        self.block1_sam = SAM(feature_num, activation, feature_num, kernel_size=(3, 3), stride=1, groups = groups, padding='same')
        self.output_channels = feature_num

    def forward(self, input):
        block1_output = self.block1_end(input)
        block1_output, block1_loss = self.block1_sam(block1_output, input)
        return block1_loss

    def getOutputChannels(self):
        return self.output_channels