import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN

class DenseNextBlock(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride, padding='same', groups=1, dilation=1, bias=True):
        super(DenseNextBlock, self).__init__()
        self.kernel_size = None
        self.branch1_list = nn.ModuleList([])
        self.branch2_list = nn.ModuleList([])
        for index in range(0, groups):
            self.branch1_list.append(Conv2D_BN(in_channels, activation, max(1, int(out_channels[0] / groups)), kernel_size=(1, 1), stride=stride[0], padding=padding, groups=1, dilation = dilation, bias=bias))
        for index in range(0, groups):
            self.branch2_list.append(Conv2D_BN(out_channels[0], activation, max(1, int(out_channels[1] / groups)), kernel_size=kernel_size, stride=stride[0], padding=padding, groups=1, dilation = dilation, bias=bias))
        self.identity = Conv2D_BN(in_channels, activation, in_channels, kernel_size=(1, 1), stride=stride[1], padding=padding, dilation = dilation, bias=bias)

    def forward(self, input):
        outputs = []
        for b in self.branch1_list:
            outputs.append(b(input))
        output = torch.cat(outputs, dim = 1)
        outputs = []
        for b in self.branch2_list:
            outputs.append(b(output))
        output = torch.cat(outputs, dim = 1)

        if output.shape[1:4] == input.shape[1:4]:
            identity = input
        else:
            identity = self.identity(input)
        output = torch.cat([output, identity], dim = 1)
        return output