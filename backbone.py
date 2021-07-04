from model.layers import *
from model.densenet import *
from model.densenext import *
from model.resnet import *
from model.resnext import *
from torchsummary import summary

if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 224, 224)
    batch_size = 1
    model = DenseNext32(activation)
    summary(model, input_shape, batch_size=batch_size, device='cpu')