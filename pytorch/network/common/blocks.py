import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN

class StemBlock(nn.Module):
    def __init__(self, in_channels, activation, bias = True):
        super(StemBlock, self).__init__()
        self.output_channel = 128
        self.conv2d_1 = Conv2D_BN(in_channels, activation=activation, out_channels=32, kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_2 = Conv2D_BN(32, activation, out_channels=64, kernel_size=(3, 3), stride=2, padding='same', bias=bias)
        self.conv2d_3 = Conv2D_BN(64, activation, out_channels=32, kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_4 = Conv2D_BN(32, activation, out_channels=64, kernel_size=(3, 3), stride=1, padding='same', bias=bias)
        self.conv2d_5 = Conv2D_BN(64, activation, out_channels=self.output_channel, kernel_size=(3, 3), stride=2, padding='same', bias=True)


    def forward(self, input):
        output1 = self.conv2d_1(input)
        output2 = self.conv2d_2(output1)
        output3 = self.conv2d_3(output2)
        output4 = self.conv2d_4(output3)
        shortcut = output2 + output4
        stemout = self.conv2d_5(shortcut)
        return stemout

    def getOutputChannel(self):
        return self.output_channel