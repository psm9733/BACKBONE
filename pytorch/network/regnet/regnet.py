from network.common.layers import *
from network.common.blocks import *
from network.regnet.blocks import *
from network.regnet.stage import RegNetStage
import torch.nn as nn

class RegNet(nn.Module):
    def __init__(self, activation, in_channels, block_width, bottleneck_ratio, groups, padding='same', dilation=1, bias=True):
        super(RegNet, self).__init__()
        self.stem = StemBlock(in_channels, activation, bias)
        self.stage1 = RegNetStage(activation, 3, self.stem.getOutputChannel(), block_width, bottleneck_ratio, groups, padding, dilation)
        self.stage2 = RegNetStage(activation, 3, block_width, block_width, bottleneck_ratio, groups, padding, dilation)
        self.stage3 = RegNetStage(activation, 3, block_width, block_width, bottleneck_ratio, groups, padding, dilation)
        self.stage4 = RegNetStage(activation, 3, block_width, block_width, bottleneck_ratio, groups, padding, dilation)
        self.body = nn.Sequential(
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
        )
        self.output_channel = block_width

    def forward(self, input):
        output = self.stem(input)
        output = self.body(output)
        return output

    def getOutputChannel(self):
        return self.output_channel