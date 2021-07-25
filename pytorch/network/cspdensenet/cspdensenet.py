from network.common.layers import *
from network.densenet.blocks import *
import torch.nn as nn

class CspDenseNetMicro(nn.Module):
    def __init__(self, activation, bias=True):
        super(CspDenseNetMicro, self).__init__()