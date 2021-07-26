import os
import tqdm
import torch
import albumentations
import albumentations.pytorch
import warnings
import torch.nn as nn
import cv2
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from utils.generator import NoiseReduction
from network.model import DeNoising
from utils.losses import PSNRLoss
from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import os

import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    model_name="SHNet"
    input_shape = (3, 256, 256)
    feature_num = 256
    batch_size = 8
    worker = 4
    max_lr = 1e-3
    min_lr = 1e-5
    weight_decay = 1e-4
    max_epochs = 200
    timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    logdir = "./logs/" + timestamp
    save_dir = "./saved_model/" + timestamp
    if os.path.isdir('./logs') == False:
        os.mkdir('./logs')
    if os.path.isdir('./saved_model') == False:
        os.mkdir('./saved_model')
    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)

    tb_logger = pl_loggers.TensorBoardLogger(logdir, name=model_name, default_hp_metric=False)
    model = DeNoising(feature_num=feature_num, min_lr=min_lr, max_lr=max_lr, weight_decay=weight_decay)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        warnings.warn('Cannot use CUDA context. Train might be slower!')
    else:
        warnings.warn('Single GPU Activate!')
    model.to(device)

    # data setup
    train_transform = albumentations.Compose([
        albumentations.Resize(height=input_shape[1], width=input_shape[2]),
        albumentations.Normalize(0, 1),
        albumentations.SomeOf([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(-90, 90),
        ], 2, p=0.5),
        albumentations.pytorch.ToTensorV2(),
    ], additional_targets={'image1': 'image', 'image2': 'image'})
    valid_transform = albumentations.Compose([
        albumentations.Resize(height=input_shape[1], width=input_shape[2]),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], additional_targets={'image1': 'image', 'image2': 'image'})

    trainLoader = DataLoader(NoiseReduction('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', True, input_shape, train_transform), batch_size=batch_size, shuffle=True, num_workers=worker, drop_last=True)
    validLoader = DataLoader(NoiseReduction('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', False, input_shape, valid_transform), batch_size=batch_size, num_workers=worker)

    pl.seed_everything(1)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename=model_name, monitor="PSNR_loss", mode='min', verbose=True, save_top_k=1)
    trainer = pl.Trainer(precision=16, max_epochs=max_epochs, gpus=1, accumulate_grad_batches = 1, logger=tb_logger, callbacks=checkpoint_callback)
    trainer.fit(model, trainLoader, validLoader)

if __name__ == "__main__":
    main()