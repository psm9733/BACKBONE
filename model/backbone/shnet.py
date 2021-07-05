from model.layers import *
from model.blocks import *

class SHNet(nn.Module):          #StackedHourGlass
    def __init__(self, activation, feature_num = 256, bias = True):
        super(SHNet, self).__init__()
        self.stemblock = StemBlock(in_channels=3, activation=activation, out_channels=(int(feature_num / 8), int(feature_num / 4)), kernel_size=(3, 3), stride=(2, 1), padding='same')
        self.downblock1 = HourglassDownBlock(in_channels=int(feature_num / 4), activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), stride=(1, 2, 1), padding='same')
        self.downblock2 = HourglassDownBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), stride=(1, 2, 1), padding='same')
        self.downblock3 = HourglassDownBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), stride=(1, 2, 1), padding='same')
        self.downblock4 = HourglassDownBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), stride=(1, 2, 1), padding='same')

        self.sameblock1 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')
        self.sameblock2 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')

        self.upblock1 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding='same', mode='nearest')
        self.upblock2 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding='same', mode='nearest')
        self.upblock3 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding='same', mode='nearest')
        self.upblock4 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding='same', mode='nearest')
        self.upblock5 = HourglassUpBlock(in_channels=feature_num, activation=activation, out_channels=(feature_num, int(feature_num / 2), feature_num), kernel_size=(3, 3), padding='same', mode='nearest')

        self.intermediateBlock1 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')
        self.intermediateBlock2 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')
        self.intermediateBlock3 = ResidualBlock(in_channels=feature_num, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')
        self.intermediateBlock4 = ResidualBlock(in_channels=int(feature_num / 4), activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), int(feature_num)), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')
        self.intermediateBlock5 = ResidualBlock(in_channels=3, activation=activation, out_channels=(int(feature_num / 2), int(feature_num / 2), feature_num), kernel_size=(3, 3), stride=(1, 1, 1), padding='same')

        self.output = Conv2D_BN(feature_num, activation=activation, out_channels=3, kernel_size=(7, 7), stride = 1, padding='same')

    def forward(self, input):
        stem_out = self.stemblock(input)
        down_out1 = self.downblock1(stem_out)
        down_out2 = self.downblock2(down_out1)
        down_out3 = self.downblock3(down_out2)
        down_out4 = self.downblock4(down_out3)

        same_out1 = self.sameblock1(down_out4)
        same_out2 = self.sameblock2(same_out1)

        intermediate_out1 = self.intermediateBlock1(down_out3)
        intermediate_out2 = self.intermediateBlock2(down_out2)
        intermediate_out3 = self.intermediateBlock3(down_out1)
        intermediate_out4 = self.intermediateBlock4(stem_out)
        intermediate_out5 = self.intermediateBlock5(input)

        up_out1 = self.upblock1(same_out2)
        up_out1 = up_out1 + intermediate_out1
        up_out2 = self.upblock2(up_out1)
        up_out2 = up_out2 + intermediate_out2
        up_out3 = self.upblock3(up_out2)
        up_out3 = up_out3 + intermediate_out3
        up_out4 = self.upblock4(up_out3)
        up_out4 = up_out4 + intermediate_out4
        heatmaps = self.upblock5(up_out4)
        heatmaps = heatmaps + intermediate_out5

        output = self.output(heatmaps)
        return output