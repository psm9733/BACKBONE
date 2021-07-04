from model.backbone.densenet import *
from model.backbone.densenext import *
from model.backbone.resnet import *
from model.backbone.resnext import *
from model.backbone.hourglass import *
from torchsummary import summary

if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 416, 416)
    batch_size = 1
    model = HourGlass(activation)
    summary(model, input_shape, batch_size=batch_size, device='cpu')