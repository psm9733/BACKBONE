import torch
import torch.nn as nn
from model.layers import Conv2D_BN
from model.backbone import ResNet50
from utils.utils import weight_initialize
from torchsummary import summary

class Classification(nn.Module):
    def __init__(self, activation, classes):
        super(Classification, self).__init__()
        self.backbone = ResNet50(activation)
        self.classification_head = nn.Sequential(
            Conv2D_BN(self.backbone.block4_end.Conv2D_BN_3.conv_layer.out_channels, activation, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, input):
        output = self.backbone(input)
        output = self.classification_head(output)
        return output

if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 224, 224)
    classes = 200
    model = Classification(activation, classes)
    weight_initialize(model)
    summary(model, input_shape, batch_size=4, device='cpu')