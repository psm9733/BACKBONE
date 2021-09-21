from network.common.layers import Conv2D_BN
import torch.nn as nn

class XBlock(nn.Module):            #based ResidualBlock + group convolution
    def __init__(self, activation, block_width, bottleneck_ratio, stride, padding = 'same', groups = 1, dilation = 1, bias = True):
        super(XBlock, self).__init__()
        self.conv2d_bn_1 = Conv2D_BN(block_width, activation, int(block_width / bottleneck_ratio), kernel_size=(1, 1), stride=stride[0], padding=padding, bias=bias)
        self.conv2d_bn_2 = Conv2D_BN(int(block_width / bottleneck_ratio), activation, int(block_width / bottleneck_ratio), kernel_size=kernel_size, stride=stride[1], padding=padding, groups=groups, bias=bias)
        self.conv2d_bn_3 = Conv2D_BN(int(block_width / bottleneck_ratio), activation, block_width, kernel_size=(1, 1), stride=stride[2], padding=padding, bias=bias)
        self.identity = Conv2D_BN(block_width, activation, block_width, kernel_size=(1, 1), stride=stride[1], padding=padding, bias=bias)

    def forward(self, input):
        output = self.conv2d_bn_1(input)
        output = self.conv2d_bn_2(output)
        output = self.conv2d_bn_3(output)
        if output.shape != input.shape:
            identity = self.identity(input)
        else:
            identity = input
        output += identity
        return output

