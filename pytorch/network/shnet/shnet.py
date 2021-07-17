import torch
from network.common.layers import *
from network.shnet.blocks import *
from network.common.modules import SAM

class SHNet(nn.Module):          #StackedHourGlass
    def __init__(self, activation, feature_num = 512, groups = 32, mode = "", bias = True):
        super(SHNet, self).__init__()
        # self.stem = StemBlock(in_channels=3, activation=activation, out_channels=(int(feature_num / 8), int(feature_num / 4)), kernel_size=(3, 3), stride=(2, 1), padding='same', bias=bias)
        self.stem = Conv2D_BN(3, activation=activation, out_channels=int(feature_num / 4), kernel_size=(7, 7), stride=1, padding='same')
        self.block1_end = HourglassBlock(activation, int(feature_num / 4), feature_num, mode, groups = groups)
        self.block1_sam = SAM(feature_num, activation, feature_num, kernel_size=(1, 1), stride=1, groups = groups, padding='same')
        self.block2_end = HourglassBlock(activation, feature_num, feature_num, mode, groups = groups)
        self.block2_sam = SAM(feature_num, activation, feature_num, kernel_size=(1, 1), stride=1, groups = groups, padding='same')
        self.block3_end = HourglassBlock(activation, feature_num, feature_num, mode, groups = groups)
        self.block3_sam = SAM(feature_num, activation, feature_num, kernel_size=(1, 1), stride=1, groups = groups, padding='same')
        self.output_channel = feature_num

    def forward(self, input):
        output = self.stem(input)
        block1_output = self.block1_end(output)
        block1_output, block1_loss = self.block1_sam(block1_output, input)

        block2_output = self.block2_end(block1_output)
        block2_output, block2_loss= self.block2_sam(block2_output, input)

        block3_output = self.block3_end(block2_output)
        block3_output, block3_loss = self.block3_sam(block3_output, input)
        return block1_loss, block2_loss, block3_loss

    def getOutputChannel(self):
        return self.output_channel