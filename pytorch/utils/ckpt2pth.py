import os
import torch
from network.model import *

def main():
    model_name="DenseNext32"
    input_shape = (1, 28, 28)
    num_classes = 10
    # PATH = "./saved_model/20210730005117/SHNet.ckpt"
    PATH = "./saved_model/20210730153513/DenseNext32.ckpt"
    # model = DeNoising(num_classes)
    model = Classification(num_classes)
    model.load_state_dict(torch.load(PATH))
    torch.save(model.state_dict(), "./resave_" + model_name + ".pth")

if __name__ == "__main__":
    main()
