import torch
import torch.nn as nn
from layers import Conv2D_BN
from resnet import *
from resnext import *
from densenet import *
from utils.utils import weight_initialize
from torchsummary import summary

class Classification(nn.Module):
    def __init__(self, activation, classes):
        super(Classification, self).__init__()
        self.backbone = ResNext14(activation)
        in_channel = self.backbone.output.Conv2D_BN_3.conv_layer.out_channels           #resnet, resnext
        # in_channel = self.backbone.output.Conv2D_BN_2.conv_layer.out_channels + self.backbone.output.Conv2D_BN_1.conv_layer.in_channels         #densenet
        self.classification_head = nn.Sequential(
            Conv2D_BN(in_channel, activation, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        output = self.backbone(input)
        output = self.classification_head(output)
        b, c, _, _ = output.size()
        output = output.view(b, c)
        return {'pred': output}

if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 224, 224)
    classes = 200
    model = Classification(activation, classes)
    weight_initialize(model)
    summary(model, input_shape, batch_size=4, device='cpu')