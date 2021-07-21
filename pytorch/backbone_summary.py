from network.densenet.densenet import *
from network.densenext.densenext import *
from network.resnet.resnet import *
from network.resnext.resnext import *
from network.shnet.shnet import *
from torchsummary import summary
from torch.autograd import Variable
from network.model import DeNoising, Classification
import torch.onnx
import torch._C as _C
TrainingMode = _C._onnx.TrainingMode

if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 256, 256)
    # input_shape = (3, 224, 224)
    batch_size = 1
    feature_num = 128
    model = DeNoising(activation, feature_num, groups=32)
    summary(model, input_shape, batch_size=batch_size, device='cpu')
    dummy_input = Variable(torch.randn(4, input_shape[0], input_shape[1], input_shape[2]))
    torch.onnx.export(model, dummy_input, "model.onnx", training=TrainingMode.TRAINING, opset_version=11)