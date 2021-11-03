from network.common.layers import *
from network.common.blocks import *
from network.regnet.blocks import *
from network.regnet.stage import RegNetXStage, RegNetYStage
import torch.nn as nn

<<<<<<< HEAD
class RegNetX_3stage(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=32, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 1, 1]
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels = [128, 256, 512]         #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
  
=======
class RegNetX_mini(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=32, padding='same', dilation=1, bias=True):
        super(RegNetX_mini, self).__init__()
        self.stage_depth = [1, 1, 1, 1]
        self.output_stride = 16
        self.output_branch_channels = [64, 128, 256, 512]         #block_width
        self.output_channels = self.output_branch_channels[3]
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
<<<<<<< HEAD
        return [stage1_out, stage2_out, stage3_out]
=======
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetX_200MF_custom(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=32, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 1, 1, 1]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels = [128, 256, 512, 1024]         #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels = [128, 256, 512, 1024]         #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetX_200MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=8, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 1, 4, 7]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels = [24, 56, 152, 368]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels = [24, 56, 152, 368]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetX_400MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=16, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 2, 7, 12]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels = [32, 64, 160, 384]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels = [32, 64, 160, 384]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetX_600MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=24, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 3, 5, 7]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels = [48, 96, 240, 528]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels = [48, 96, 240, 528]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetX_800MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=16, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 3, 7, 5]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels = [64, 128, 288, 672]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels = [64, 128, 288, 672]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetX_1600MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=24, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [2, 4, 10, 2]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[72, 168, 408, 912]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[72, 168, 408, 912]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetX_3200MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=48, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [2, 6, 15, 2]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[96, 192, 432, 1008]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[96, 192, 432, 1008]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetX_4000MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=40, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [2, 5, 14, 2]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[80, 240, 560, 1360]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[80, 240, 560, 1360]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetX_6400MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=56, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [2, 4, 10, 1]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[168, 392, 784, 1624]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[168, 392, 784, 1624]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetX_8000MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=120, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [2, 5, 15, 1]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[80, 240, 720, 1920]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[80, 240, 720, 1920]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetXStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetXStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetXStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetXStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

<<<<<<< HEAD
class RegNetY_3stage(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=32, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 1, 1]
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels = [128, 256, 512]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
=======
class RegNetY_mini(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=32, padding='same', dilation=1, bias=True):
        super(RegNetY_mini, self).__init__()
        self.stage_depth = [1, 1, 1, 1]
        self.output_stride = 16
        self.output_branch_channels = [64, 128, 256, 512]        #block_width
        self.output_channels = self.output_branch_channels[3]
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
<<<<<<< HEAD
        return [stage1_out, stage2_out, stage3_out]
=======
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetY_200MF_custom(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=32, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 1, 1, 1]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[128, 256, 512, 1024]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[128, 256, 512, 1024]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetY_200MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=8, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 1, 4, 7]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[24, 56, 152, 368]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[24, 56, 152, 368]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetY_400MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=16, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 2, 7, 12]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[32, 64, 160, 384]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[32, 64, 160, 384]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetY_600MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=24, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 3, 5, 7]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[48, 96, 240, 528]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[48, 96, 240, 528]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetY_800MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=16, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [1, 3, 7, 5]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[64, 128, 288, 672]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[64, 128, 288, 672]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetY_1600MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=24, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [2, 4, 10, 2]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[72, 168, 408, 912]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[72, 168, 408, 912]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetY_3200MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=48, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [2, 6, 15, 2]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[96, 192, 432, 1008]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[96, 192, 432, 1008]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetY_4000MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=40, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [2, 5, 14, 2]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[80, 240, 560, 1360]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[80, 240, 560, 1360]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetY_6400MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=56, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [2, 4, 10, 1]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[168, 392, 784, 1624]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[168, 392, 784, 1624]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride

class RegNetY_8000MF(nn.Module):
    def __init__(self, activation, in_channels, bottleneck_ratio=1, groups=120, padding='same', dilation=1, bias=True):
        super().__init__()
        self.stage_depth = [2, 5, 15, 1]
<<<<<<< HEAD
        self.output_stride = 2**len(self.stage_depth)
        self.output_branch_channels=[80, 240, 720, 1920]        #block_width
        self.output_channels = self.output_branch_channels[len(self.output_branch_channels) - 1]
=======
        self.output_stride = 16
        self.output_branch_channels=[80, 240, 720, 1920]        #block_width
        self.output_channels = self.output_branch_channels[3]
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.stage1 = RegNetYStage(activation, self.stage_depth[0], in_channels, self.output_branch_channels[0], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage2 = RegNetYStage(activation, self.stage_depth[1], self.output_branch_channels[0], self.output_branch_channels[1], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage3 = RegNetYStage(activation, self.stage_depth[2], self.output_branch_channels[1], self.output_branch_channels[2], bottleneck_ratio, groups, padding, dilation, bias)
        self.stage4 = RegNetYStage(activation, self.stage_depth[3], self.output_branch_channels[2], self.output_branch_channels[3], bottleneck_ratio, groups, padding, dilation, bias)

    def forward(self, input):
        stage1_out = self.stage1(input)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        return [stage1_out, stage2_out, stage3_out, stage4_out]

    def getOutputChannels(self):
        return self.output_channels

    def getOutputBranchChannels(self):
        return self.output_branch_channels

    def getOutputStride(self):
        return self.output_stride