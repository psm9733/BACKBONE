from model.layers import *
from model.blocks import *

class SHNet(nn.Module):          #StackedHourGlass
    def __init__(self, activation, feature_num = 256, bias = True):
        super(SHNet, self).__init__()
        self.stem = StemBlock(in_channels=3, activation=activation, out_channels=(int(feature_num / 8), int(feature_num / 4)), kernel_size=(3, 3), stride=(2, 1), padding='same')
        self.block1_end = HourglassBlock(activation, feature_num)

    def forward(self, input):
        output = self.stem(input)
        output = self.block1_end(output)
        return output

class SHNetTiny(nn.Module):          #StackedHourGlass
    def __init__(self, activation, feature_num = 256, bias = True):
        super(SHNetTiny, self).__init__()
        self.stem = self.stemblock = StemBlock(in_channels=3, activation=activation, out_channels=(int(feature_num / 8), int(feature_num / 4)), kernel_size=(3, 3), stride=(2, 1), padding='same')
        self.block1_end = HourglassBlockTiny(activation, feature_num)

    def forward(self, input):
        output = self.stem(input)
        output = self.block1_end(input)
        return output

class SHNetMicro(nn.Module):          #StackedHourGlass
    def __init__(self, activation, feature_num = 256, bias = True):
        super(SHNetMicro, self).__init__()
        self.stem = self.stemblock = StemBlock(in_channels=3, activation=activation, out_channels=(int(feature_num / 8), int(feature_num / 4)), kernel_size=(3, 3), stride=(2, 1), padding='same')
        self.block1_end = HourglassBlockMicro(activation, feature_num)

    def forward(self, input):
        output = self.stem(input)
        output = self.block1_end(input)
        return output