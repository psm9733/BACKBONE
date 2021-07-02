from model.layers import *

class ResNet26(nn.Module):
    def __init__(self, activation, bias = True):
        super(ResNet26, self).__init__()
        self.block1_end = ResidualBlock(in_channels=128, activation=activation, out_channels=(32, 32, 128), kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.block2_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=2, padding=(1, 1))
        self.block3_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.block4_end = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding=(1, 1))

        self.resnet26 = nn.Sequential(
            Conv2D_BN(3, activation=activation, out_channels=64, kernel_size=(7, 7), stride=2, padding=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),
            ResidualBlock(in_channels=64, activation=activation, out_channels=(32, 32, 128), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            self.block1_end,

            ResidualBlock(in_channels=128, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            self.block2_end,

            ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            self.block3_end,

            ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            self.block4_end
        )
    def forward(self, input):
        output = self.resnet26(input)
        return output

class ResNet50(nn.Module):
    def __init__(self, activation, bias = True):
        super(ResNet50, self).__init__()
        self.block1_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.block2_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.block3_end = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.block4_end = ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=1, padding=(1, 1))

        self.resnet50 = nn.Sequential(
            Conv2D_BN(3, activation=activation, out_channels=64, kernel_size=(7, 7), stride = 2, padding=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),

            ResidualBlock(in_channels=64, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            self.block1_end,

            ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            self.block2_end,

            ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            self.block3_end,

            ResidualBlock(in_channels=1024, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            self.block4_end
        )

    def forward(self, input):
        output = self.resnet50(input)
        return output

class DenseNet64(nn.Module):
    def __init__(self, activation, bias=True):
        super(DenseNet64, self).__init__()
        self.block1_end = DenseBlock(in_channels=192, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
        self.block2_end = DenseBlock(in_channels=384, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
        self.block3_end = DenseBlock(in_channels=672, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
        self.block4_end = DenseBlock(in_channels=1024, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
        self.resnet121 = nn.Sequential(
            Conv2D_BN(3, activation=activation, out_channels=64, kernel_size=(7, 7), stride=2, padding=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),

            # block1
            DenseBlock(in_channels=64, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=96, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=128, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=160, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            self.block1_end,

            # block2
            DenseBlock(in_channels=224, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            DenseBlock(in_channels=256, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=288, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=320, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=352, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            self.block2_end,

            # block3
            DenseBlock(in_channels=416, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=448, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            DenseBlock(in_channels=480, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=512, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=544, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=576, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=608, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=640, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            self.block3_end,

            # block4
            DenseBlock(in_channels=704, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            DenseBlock(in_channels=736, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=768, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=800, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=832, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=864, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=896, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=928, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=960, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            DenseBlock(in_channels=992, activation=activation, out_channels=(128, 32), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            self.block4_end
        )

    def forward(self, input):
        output = self.resnet50(input)
        return output

if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 224, 224)
    model = ResNet50(activation)