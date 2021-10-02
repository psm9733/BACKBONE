from network.common.layers import *
from network.common.blocks import *
from network.regnet.blocks import *
from network.regnet.stage import RegNetXStage, RegNetYStage
import torch.nn as nn

class RegNetX_200MF_custom(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=32, padding='same', dilation=1, bias=True):
        super(RegNetX_200MF_custom, self).__init__()
        self.stage_depth = [1, 1, 1, 1]
        self.block_width=[128, 256, 512, 1024]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetX_200MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=8, padding='same', dilation=1, bias=True):
        super(RegNetX_200MF, self).__init__()
        self.stage_depth = [1, 1, 4, 7]
        self.block_width=[24, 56, 152, 368]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetX_400MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=16, padding='same', dilation=1, bias=True):
        super(RegNetX_400MF, self).__init__()
        self.stage_depth = [1, 2, 7, 12]
        self.block_width=[32, 64, 160, 384]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetX_600MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=24, padding='same', dilation=1, bias=True):
        super(RegNetX_600MF, self).__init__()
        self.stage_depth = [1, 3, 5, 7]
        self.block_width=[48, 96, 240, 528]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetX_800MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=16, padding='same', dilation=1, bias=True):
        super(RegNetX_800MF, self).__init__()
        self.stage_depth = [1, 3, 7, 5]
        self.block_width=[64, 128, 288, 672]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetX_1600MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=24, padding='same', dilation=1, bias=True):
        super(RegNetX_1600MF, self).__init__()
        self.stage_depth = [2, 4, 10, 2]
        self.block_width=[72, 168, 408, 912]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetX_3200MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=48, padding='same', dilation=1, bias=True):
        super(RegNetX_3200MF, self).__init__()
        self.stage_depth = [2, 6, 15, 2]
        self.block_width=[96, 192, 432, 1008]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetX_4000MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=40, padding='same', dilation=1, bias=True):
        super(RegNetX_4000MF, self).__init__()
        self.stage_depth = [2, 5, 14, 2]
        self.block_width=[80, 240, 560, 1360]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetX_6400MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=56, padding='same', dilation=1, bias=True):
        super(RegNetX_6400MF, self).__init__()
        self.stage_depth = [2, 4, 10, 1]
        self.block_width=[168, 392, 784, 1624]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetX_8000MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=120, padding='same', dilation=1, bias=True):
        super(RegNetX_8000MF, self).__init__()
        self.stage_depth = [2, 5, 15, 1]
        self.block_width=[80, 240, 720, 1920]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetY_200MF_custom(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=32, padding='same', dilation=1, bias=True):
        super(RegNetY_200MF_custom, self).__init__()
        self.stage_depth = [1, 1, 1, 1]
        self.block_width=[128, 256, 512, 1024]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetY_200MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=8, padding='same', dilation=1, bias=True):
        super(RegNetY_200MF, self).__init__()
        self.stage_depth = [1, 1, 4, 7]
        self.block_width=[24, 56, 152, 368]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetY_400MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=16, padding='same', dilation=1, bias=True):
        super(RegNetY_400MF, self).__init__()
        self.stage_depth = [1, 2, 7, 12]
        self.block_width=[32, 64, 160, 384]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetY_600MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=24, padding='same', dilation=1, bias=True):
        super(RegNetY_600MF, self).__init__()
        self.stage_depth = [1, 3, 5, 7]
        self.block_width=[48, 96, 240, 528]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetY_800MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=16, padding='same', dilation=1, bias=True):
        super(RegNetY_800MF, self).__init__()
        self.stage_depth = [1, 3, 7, 5]
        self.block_width=[64, 128, 288, 672]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetY_1600MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=24, padding='same', dilation=1, bias=True):
        super(RegNetY_1600MF, self).__init__()
        self.stage_depth = [2, 4, 10, 2]
        self.block_width=[72, 168, 408, 912]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetY_3200MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=48, padding='same', dilation=1, bias=True):
        super(RegNetY_3200MF, self).__init__()
        self.stage_depth = [2, 6, 15, 2]
        self.block_width=[96, 192, 432, 1008]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

class RegNetY_4000MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=40, padding='same', dilation=1, bias=True):
        super(RegNetY_4000MF, self).__init__()
        self.stage_depth = [2, 5, 14, 2]
        self.block_width=[80, 240, 560, 1360]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetY_6400MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=56, padding='same', dilation=1, bias=True):
        super(RegNetY_6400MF, self).__init__()
        self.stage_depth = [2, 4, 10, 1]
        self.block_width=[168, 392, 784, 1624]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride

class RegNetY_8000MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=120, padding='same', dilation=1, bias=True):
        super(RegNetY_8000MF, self).__init__()
        self.stage_depth = [2, 5, 15, 1]
        self.block_width=[80, 240, 720, 1920]
        self.output_stride = 16
        self.output_channel = self.block_width[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.block_width[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.block_width[0], self.block_width[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.block_width[1], self.block_width[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.block_width[2], self.block_width[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannel(self):
        return self.output_channel

    def getOutputStride(self):
        return self.output_stride