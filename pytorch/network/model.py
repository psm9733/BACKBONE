import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN
from network.resnet.resnet import *
from network.resnext.resnext import *
from network.densenet.densenet import *
from network.densenext.densenext import *
from network.shnet.shnet import *
from network.unet.unet import *
from utils.utils import weight_initialize
from torchsummary import summary

class Classification(nn.Module):
    def __init__(self, activation, classes):
        super(Classification, self).__init__()
        self.backbone = DenseNext64(activation)
        self.classification_head = nn.Sequential(
            Conv2D_BN(self.backbone.getOutputChannel(), activation, 1280, (1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        output = self.backbone(input)
        output = self.classification_head(output)
        b, c, _, _ = output.size()
        output = output.view(b, c)
        return {'pred': output}

class DeNoising(nn.Module):
    def __init__(self, activation, feature_num):
        super(DeNoising, self).__init__()
        self.backbone = SHNet(activation, feature_num, mode = "")
        self.segmentation_head = Conv2D_BN(self.backbone.getOutputChannel(), None, 3, (1, 1))
        # self.backbone = UNet(n_channels=3, n_classes=3, bilinear=True)

    def forward(self, input):
        output = self.backbone(input)
        output = self.segmentation_head(output)
        output += input
        return {'pred': output}

class Yolov4Micro(nn.Module):
    def __init__(self, activation, classes):
        super(Yolov4Micro, self).__init__()

    def forward(self, input):
        return input


if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 224, 224)
    classes = 200
    model = Classification(activation, classes)
    weight_initialize(model)
    summary(model, input_shape, batch_size=4, device='cpu')