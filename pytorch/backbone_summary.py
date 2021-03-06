from network.densenet.densenet import *
from network.densenext.densenext import *
from network.resnet.resnet import *
from network.resnext.resnext import *
from network.hourglassnet.hourglassnet import *
from torchsummary import summary
from torch.autograd import Variable
from classification.model import Classification
from denoising.model import DeNoising
from gaze_estimation.model import E3GazeNet
from detection.model import Scaled_Yolov4
import torch.onnx
import torch._C as _C
TrainingMode = _C._onnx.TrainingMode

def model_save_onnx(model, dummy_input, name, verbose = True):
    print("=================== Saving {} model ===================".format(name))
    if verbose:
        summary(model, input_shape, batch_size=batch_size, device='cpu')
    torch.onnx.export(model, dummy_input, name + ".onnx", training=TrainingMode.TRAINING, opset_version=11)

if __name__ == "__main__":
    activation = nn.ReLU()
    class_num = 91
    batch_size = 4
    input_shape = (3, 512, 512)
    dummy_input = torch.tensor(torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]))
    model = Classification("tiny_imagenet")
    model.eval()
    model_save_onnx(model, dummy_input, 'Classification')

    input_shape = (3, 512, 512)
    dummy_input = torch.tensor(torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]))
    model = DeNoising(128, input_shape)
    model.eval()
    model_save_onnx(model, dummy_input, 'DeNoising')

    input_shape = (1, 128, 128)
    dummy_input = torch.tensor(torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]))
    model = E3GazeNet(input_shape)
    model.eval()
    model_save_onnx(model, dummy_input, 'E3GazeNet')

    input_shape = (3, 512, 512)
    dummy_input = torch.tensor(torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2]))

    model = Scaled_Yolov4(input_shape, class_num)
    model.eval()
    model_save_onnx(model, dummy_input, 'Scaled_Yolov4')
