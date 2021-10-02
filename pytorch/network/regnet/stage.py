from network.regnet.blocks import XBlock, YBlock
import torch.nn as nn

class RegNetXStage(nn.Module):
    def __init__(self, activation, block_num, in_channels, block_width, bottleneck_ratio, groups=1, padding='same', dilation=1, bias=True):
        super(RegNetXStage, self).__init__()
        self.stage = nn.ModuleList([])
        for index in range(0, block_num):
            if index == 0:
                self.stage.append(XBlock(activation, in_channels, block_width, bottleneck_ratio, 2, padding, groups, dilation, bias))
            else:
                self.stage.append(XBlock(activation, block_width, block_width, bottleneck_ratio, 1, padding, groups, dilation, bias))

    def forward(self, input):
        output = input
        for xblock in self.stage:
            output = xblock(output)
        return output

class RegNetYStage(nn.Module):
    def __init__(self, activation, block_num, in_channels, block_width, bottleneck_ratio, groups=1, padding='same', dilation=1, bias=True):
        super(RegNetYStage, self).__init__()
        self.stage = nn.ModuleList([])
        for index in range(0, block_num):
            if index == 0:
                self.stage.append(YBlock(activation, in_channels, block_width, bottleneck_ratio, 2, padding, groups, dilation, bias))
            else:
                self.stage.append(YBlock(activation, block_width, block_width, bottleneck_ratio, 1, padding, groups, dilation, bias))

    def forward(self, input):
        output = input
        for xblock in self.stage:
            output = xblock(output)
        return output

