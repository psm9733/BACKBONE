import torch.nn as nn
from network.common.layers import Conv2D_BN

class Yolo_3branch(nn.Module):
    def __init__(self, in_branch_channels, out_branch_channels, bias=True):
        super().__init__()
        self.conv2d_linear_branch1 = Conv2D_BN(in_branch_channels[0], activation=None, out_channels=out_branch_channels[0], kernel_size=(1, 1), stride=1, padding='same', bias=bias)
        self.conv2d_linear_branch2 = Conv2D_BN(in_branch_channels[1], activation=None, out_channels=out_branch_channels[1], kernel_size=(1, 1), stride=1, padding='same', bias=bias)
        self.conv2d_linear_branch3 = Conv2D_BN(in_branch_channels[2], activation=None, out_channels=out_branch_channels[2], kernel_size=(1, 1), stride=1, padding='same', bias=bias)

    def forward(self, x1, x2, x3):
        big_out = self.conv2d_linear_branch1(x1)
        middle_out = self.conv2d_linear_branch2(x2)
        small_out = self.conv2d_linear_branch3(x3)
        return [big_out, middle_out, small_out]
