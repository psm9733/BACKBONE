from model.layers import *
from model.blocks import *

class ResNext14(nn.Module):
    def __init__(self, activation, groups, bias = True):
        super(ResNext14, self).__init__()
        self.block1_end = ResidualBlock(in_channels=64, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding='same', groups=groups)
        self.block2_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=2, padding='same', groups=groups)
        self.block3_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding='same', groups=groups)
        self.output = ResidualBlock(in_channels=1024, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=1, padding='same', groups=groups)

        self.resnext14 = nn.Sequential(
            Conv2D_BN(3, activation=activation, out_channels=64, kernel_size=(7, 7), stride=2, padding=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),
            self.block1_end,
            self.block2_end,
            self.block3_end,
            self.output
        )

    def forward(self, input):
        output = self.resnext14(input)
        return output

class ResNext26(nn.Module):
    def __init__(self, activation, groups, bias = True):
        super(ResNext26, self).__init__()
        self.block1_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding='same', groups=groups)
        self.block2_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=2, padding='same', groups=groups)
        self.block3_end = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding='same', groups=groups)
        self.output = ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=1, padding='same', groups=groups)

        self.resnext26 = nn.Sequential(
            Conv2D_BN(3, activation=activation, out_channels=64, kernel_size=(7, 7), stride=2, padding=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),
            ResidualBlock(in_channels=64, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding='same', groups=groups),
            self.block1_end,

            ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=2, padding='same', groups=groups),
            self.block2_end,

            ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=2, padding='same', groups=groups),
            self.block3_end,

            ResidualBlock(in_channels=1024, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=2, padding='same', groups=groups),
            self.output
        )

    def forward(self, input):
        output = self.resnext26(input)
        return output

class ResNext50(nn.Module):
    def __init__(self, activation, groups = 32, bias = True):
        super(ResNext50, self).__init__()
        self.block1_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding='same', groups=groups)
        self.block2_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=1, padding='same', groups=groups)
        self.block3_end = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding='same', groups=groups)
        self.output = ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=1, padding='same', groups=groups)

        self.resnext50 = nn.Sequential(
            Conv2D_BN(3, activation=activation, out_channels=64, kernel_size=(7, 7), stride = 2, padding=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),

            ResidualBlock(in_channels=64, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding='same', groups=groups),
            ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding='same', groups=groups),
            self.block1_end,

            ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=2, padding='same', groups=groups),
            ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=1, padding='same', groups=groups),
            ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=1, padding='same', groups=groups),
            self.block2_end,

            ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=2, padding='same', groups=groups),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding='same', groups=groups),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding='same', groups=groups),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding='same', groups=groups),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding='same', groups=groups),
            self.block3_end,

            ResidualBlock(in_channels=1024, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=2, padding='same', groups=groups),
            ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=1, padding='same', groups=groups),
            self.output
        )

    def forward(self, input):
        output = self.resnext50(input)
        return output