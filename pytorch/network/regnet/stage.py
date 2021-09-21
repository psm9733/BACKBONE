from network.regnet.blocks import XBlock
import torch.nn as nn

class RegNetStage(nn.Module):
    def __init__(self, activation, block_num, block_width, bottleneck_ratio, groups = 1, padding = 'same', dilation = 1, bias = True):
        super(RegNetStage, self).__init__()
        self.stride = 1
        self.stage = nn.ModuleList([])
        for index in range(0, block_num):
            if index == 0:
                self.stride = 2
            else:
                self.stride = 1
            self.stage.append(XBlock(activation, block_width, bottleneck_ratio, self.stride, padding, groups, dilation))

    def forward(self, input):
        output = input
        for xblock in self.stage:
            output = xblock(output)
        return output
