from model.layers import *
from model.blocks import *

class HourGlass(nn.Module):
    def __init__(self, activation, bias = True):
        super(HourGlass, self).__init__()
        self.hourglass = nn.Sequential(
            Conv2D_BN(3, activation=activation, out_channels=64, kernel_size=(7, 7), stride=2, padding=(3, 3)),
            HourglassDownBlock(in_channels=64, activation=activation, out_channels=(64, 64, 128), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            HourglassDownBlock(in_channels=128, activation=activation, out_channels=(128, 128, 256), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            HourglassDownBlock(in_channels=256, activation=activation, out_channels=(256, 256, 512), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            HourglassDownBlock(in_channels=512, activation=activation, out_channels=(512, 512, 1024), kernel_size=(3, 3), stride=2, padding=(1, 1)),

            HourglassUpBlock(in_channels=1024, activation=activation, out_channels=(1024, 512, 512), kernel_size=(3, 3), padding=(1, 1), bilinear=True),
            HourglassUpBlock(in_channels=512, activation=activation, out_channels=(512, 256, 256), kernel_size=(3, 3), padding=(1, 1), bilinear=True),
            HourglassUpBlock(in_channels=256, activation=activation, out_channels=(256, 256, 128), kernel_size=(3, 3), padding=(1, 1), bilinear=True),
            HourglassUpBlock(in_channels=128, activation=activation, out_channels=(128, 128, 64), kernel_size=(3, 3), padding=(1, 1), bilinear=True),
            HourglassUpBlock(in_channels=64, activation=activation, out_channels=(64, 64, 32), kernel_size=(3, 3), padding=(1, 1), bilinear=True),
        )

    def forward(self, input):
        output = self.hourglass(input)
        return output