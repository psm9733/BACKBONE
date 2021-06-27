import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Conv2D_BN(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'):
        super(Conv2D_BN, self).__init__()
        in_channels = math.floor(in_channels)
        out_channels = math.floor(out_channels)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias, padding_mode = padding_mode)
        self.batchNorm_layer = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, input):
        output = self.conv_layer(input)
        output = self.batchNorm_layer(output)
        output = self.activation(output)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'):
        super(ResidualBlock, self).__init__()
        self.Conv2D_BN_1 = Conv2D_BN(in_channels, activation, out_channels[0], kernel_size = (1, 1), stride = (1, 1), padding = 0)
        self.Conv2D_BN_2 = Conv2D_BN(out_channels[0], activation, out_channels[1], kernel_size = kernel_size, stride = stride, padding = padding)
        self.Conv2D_BN_3 = Conv2D_BN(out_channels[1], activation, out_channels[2], kernel_size = (1, 1), stride = (1, 1), padding = 0)
    
    def forward(self, input):
        output = self.Conv2D_BN_1(input)
        output = self.Conv2D_BN_2(output)
        output = self.Conv2D_BN_3(output)
        if output.shape[1] != input.shape[1]:
            input = Conv2D_BN(input.shape[1], activation, output.shape[1], kernel_size = (1, 1), stride = (1, 1), padding = 0)(input)
        output += input
        return output

if __name__ == "__main__":
    activation = nn.ReLU()

    input = torch.randn(16, 3, 224, 224)

    # resnet 50
    stem = Conv2D_BN(input.shape[1], activation = activation, out_channels = 64, kernel_size = (7, 7), stride = 2, padding = (3, 3))(input)
    stem_maxpool = nn.MaxPool2d((2, 2))(stem)

    ResBlock1_1 = ResidualBlock(in_channels = 64, activation = activation, out_channels = (64, 64, 256), kernel_size = (3, 3), stride = 1, padding = (1, 1))(stem_maxpool)
    ResBlock1_2 = ResidualBlock(in_channels = 256, activation = activation, out_channels = (64, 64, 256), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock1_1)
    ResBlock1_3 = ResidualBlock(in_channels = 256, activation = activation, out_channels = (64, 64, 256), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock1_2)

    ResBlock2_1 = ResidualBlock(in_channels = 256, activation = activation, out_channels = (128, 128, 512), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock1_3)
    ResBlock2_2 = ResidualBlock(in_channels = 512, activation = activation, out_channels = (128, 128, 512), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock2_1)
    ResBlock2_3 = ResidualBlock(in_channels = 512, activation = activation, out_channels = (128, 128, 512), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock2_2)
    ResBlock2_4 = ResidualBlock(in_channels = 512, activation = activation, out_channels = (128, 128, 512), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock2_3)

    ResBlock3_1 = ResidualBlock(in_channels = 512, activation = activation, out_channels = (256, 256, 1024), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock2_4)
    ResBlock3_2 = ResidualBlock(in_channels = 1024, activation = activation, out_channels = (256, 256, 1024), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock3_1)
    ResBlock3_3 = ResidualBlock(in_channels = 1024, activation = activation, out_channels = (256, 256, 1024), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock3_2)
    ResBlock3_4 = ResidualBlock(in_channels = 1024, activation = activation, out_channels = (256, 256, 1024), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock3_3)
    ResBlock3_5 = ResidualBlock(in_channels = 1024, activation = activation, out_channels = (256, 256, 1024), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock3_4)
    ResBlock3_6 = ResidualBlock(in_channels = 1024, activation = activation, out_channels = (256, 256, 1024), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock3_5)

    ResBlock4_1 = ResidualBlock(in_channels = 1024, activation = activation, out_channels = (512, 512, 2048), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock3_6)
    ResBlock4_2 = ResidualBlock(in_channels = 2048, activation = activation, out_channels = (512, 512, 2048), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock4_1)
    ResBlock4_3 = ResidualBlock(in_channels = 2048, activation = activation, out_channels = (512, 512, 2048), kernel_size = (3, 3), stride = 1, padding = (1, 1))(ResBlock4_2)


    model = nn.Sequential(stem, stem_maxpool, ResBlock1_1, ResBlock1_2, ResBlock1_3)
    summary(model, (3, 224, 224), batch_size=16, device='gpu')


