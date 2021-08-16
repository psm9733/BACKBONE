import torch.nn as nn
import glob
import natsort
import numpy as np

def weight_initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+divisor/2)//divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def getPadding(kernal_size, mode = 'same'):
    if mode == 'same':
        return (int((kernal_size[0] - 1) / 2), (int((kernal_size[1] - 1) / 2)))
    else:
        return 0

def splitDataset(root_dir, train_ratio):
    dataset = glob.glob(root_dir + "/**/*.jpg", recursive=True)
    dataset = natsort.natsorted(dataset)
    train_num = int(len(dataset) * train_ratio)
    train_list = dataset[0:train_num]
    valid_list = dataset[train_num:]
    return [train_list, valid_list]

def unityeye_json_process(img, json_list):
    ldmks = [eval(s) for s in json_list]
    return np.array([(x, img.shape[0]-y, z) for (x,y,z) in ldmks])