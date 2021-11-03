import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN
from utils.utils import getPadding
import torch.nn as nn

class SAM(nn.Module):
    def __init__(self, activation, in_channels, out_channels, feature_num, kernel_size, stride=1, padding='same', groups=1, dilation=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv2d_BN1 = Conv2D_BN(in_channels, activation, out_channels, kernel_size=kernel_size, stride = stride, padding=padding)
        self.conv2d_BN2 = Conv2D_BN(in_channels, None, feature_num, kernel_size=kernel_size, stride = stride, padding=padding, groups = groups)
        self.conv2d_BN3 = Conv2D_BN(out_channels, nn.Sigmoid(), feature_num, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input, origin_img):
        output1 = self.conv2d_BN1(input)
        loss_output = output1 + origin_img

        output2 = self.conv2d_BN2(input)
        output3 = self.conv2d_BN3(loss_output)
        output = torch.mul(output2, output3)
        output = output + input
        return output, loss_output
