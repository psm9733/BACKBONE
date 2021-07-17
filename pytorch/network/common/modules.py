import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN
from utils.utils import getPadding

class SAM(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=True, padding_mode='zeros'):
        super(SAM, self).__init__()
        self.padding = getPadding(kernel_size, padding)
        self.conv2d_BN1 = Conv2D_BN(in_channels, activation, 3, kernel_size=kernel_size, stride = stride, padding=self.padding)
        self.conv2d_BN2 = Conv2D_BN(in_channels, activation, out_channels, kernel_size=kernel_size, stride = stride, padding=self.padding, groups = groups)
        self.conv2d_BN3 = Conv2D_BN(3, nn.Sigmoid(), out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding)

    def forward(self, input, origin_img):
        output1 = self.conv2d_BN1(input)
        loss_output = output1 + origin_img

        output2 = self.conv2d_BN2(input)
        output3 = self.conv2d_BN3(loss_output)
        output = torch.mul(output2, output3)
        output = output + input
        return output, loss_output