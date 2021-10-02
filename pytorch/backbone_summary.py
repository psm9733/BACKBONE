from network.densenet.densenet import *
from network.densenext.densenext import *
from network.resnet.resnet import *
from network.resnext.resnext import *
from network.hourglassnet.hourglassnet import *
from torchsummary import summary
from torch.autograd import Variable
from network.model import DeNoising, Classification, E3GazeNet, Scaled_Yolov4
import torch.onnx
import torch._C as _C
TrainingMode = _C._onnx.TrainingMode

if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 416, 416)
    class_num = 91
    batch_size = 4
    model = Scaled_Yolov4(input_shape, class_num)
    model.eval()
    summary(model, input_shape, batch_size=batch_size, device='cpu')
    dummy_input = torch.tensor(torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]))
    torch.onnx.export(model, dummy_input, "model.onnx", training=TrainingMode.TRAINING, opset_version=11)