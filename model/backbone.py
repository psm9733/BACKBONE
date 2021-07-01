from model.layers import *

class ResNet26(nn.Module):
    def __init__(self, activation, bias = True):
        super(ResNet26, self).__init__()
        self.block1_end = ResidualBlock(in_channels=256, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding=(1w, 1))
        self.block2_end = ResidualBlock(in_channels=512, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.block3_end = ResidualBlock(in_channels=1024, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.block4_end = ResidualBlock(in_channels=2048, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=1, padding=(1, 1))

        self.resnet26 = nn.Sequential(
            Conv2D_BN(3, activation=activation, out_channels=64, kernel_size=(7, 7), stride=2, padding=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),
            ResidualBlock(in_channels=64, activation=activation, out_channels=(64, 64, 256), kernel_size=(3, 3), stride=1, padding=(1, 1)),
            self.block1_end,

            ResidualBlock(in_channels=256, activation=activation, out_channels=(128, 128, 512), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            self.block2_end,

            ResidualBlock(in_channels=512, activation=activation, out_channels=(256, 256, 1024), kernel_size=(3, 3), stride=2, padding=(1, 1)),
            self.block3_end,

            ResidualBlock(in_channels=1024, activation=activation, out_channels=(512, 512, 2048), kernel_size=(3, 3), stride=2, padding=(1, 1)),
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

if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 224, 224)
    model = ResNet50(activation)