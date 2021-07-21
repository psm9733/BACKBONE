from network.common.layers import *
from network.densenet.blocks import *

class CspDenseNetMicro(nn.Module):
    def __init__(self, activation, bias=True):
        super(CspDenseNetMicro, self).__init__()