from model.layers import *
from model.blocks import *

class SHNet(nn.Module):          #StackedHourGlass
    def __init__(self, activation, feature_num = 256, bias = True):
        super(SHNet, self).__init__()
        self.block1_end = HourglassBlock(activation, feature_num)

    def forward(self, input):
        output = self.block1_end(input)
        return output

class SHNetTiny(nn.Module):          #StackedHourGlass
    def __init__(self, activation, feature_num = 256, bias = True):
        super(SHNetTiny, self).__init__()
        self.block1_end = HourglassBlockTiny(activation, feature_num)

    def forward(self, input):
        output = self.block1_end(input)
        return output