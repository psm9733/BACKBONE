from network.common.layers import *
from network.common.blocks import *
from network.regnet.blocks import *
from network.regnet.stage import RegNetStage
import torch.nn as nn

class RegNet(nn.Module):
    def __init__(self, activation, in_channels, block_width, bottleneck_ratio, groups, padding='same', dilation=1, bias=True):
        super(RegNet, self).__init__()
        self.stage1 = RegNetStage(activation, 3, in_channels, block_width, bottleneck_ratio, groups, padding, dilation)
        self.stage2 = RegNetStage(activation, 3, block_width, block_width, bottleneck_ratio, groups, padding, dilation)
        self.stage3 = RegNetStage(activation, 3, block_width, block_width, bottleneck_ratio, groups, padding, dilation)
        self.stage4 = RegNetStage(activation, 3, block_width, block_width, bottleneck_ratio, groups, padding, dilation)
        self.output_channel = block_width

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel