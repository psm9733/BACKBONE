from model.backbone.densenet import *
from model.backbone.densenext import *
from model.backbone.resnet import *
from model.backbone.resnext import *
from model.backbone.shnet import *
from torchsummary import summary
from torch.autograd import Variable
from model.model import Segmentation
import torch.onnx
import torch._C as _C
TrainingMode = _C._onnx.TrainingMode

if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 384, 512)
    batch_size = 1
    feature_num = 128
    model = Segmentation(activation, feature_num)
    summary(model, input_shape, batch_size=batch_size, device='cpu')
    dummy_input = Variable(torch.randn(4, input_shape[0], input_shape[1], input_shape[2]))
    torch.onnx.export(model, dummy_input, "model.onnx", training=TrainingMode.TRAINING)