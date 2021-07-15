from network.common.layers import *
from network.shnet.blocks import *

class SHNet(nn.Module):          #StackedHourGlass
    def __init__(self, activation, feature_num = 256, mode = "", bias = True):
        super(SHNet, self).__init__()
        self.stem = StemBlock(in_channels=3, activation=activation, out_channels=(int(feature_num / 8), int(feature_num / 4)), kernel_size=(3, 3), stride=(2, 1), padding='same', bias=bias)
        self.block1_end = HourglassBlock(activation, int(feature_num / 4), feature_num, mode)
        self.output_channel = feature_num

    def forward(self, input):
        output = self.stem(input)
        output = self.block1_end(output)
        return output

    def getOutputChannel(self):
        return self.output_channel