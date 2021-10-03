import torch
from network.common.layers import *
from network.hourglassnet.blocks import *
from network.common.modules import SAM
import torch.nn as nn

class HourglassNet(nn.Module):
    def __init__(self, activation, feature_num = 512, groups = 1, mode = "", bias = True):
        super(HourglassNet, self).__init__()
        self.module = HourglassModule(activation, feature_num, mode, padding='same', groups = groups)
        self.output_channels = feature_num

    def forward(self, input):
        output = self.module(input)
        return output

    def getOutputChannels(self):
        return self.output_channels