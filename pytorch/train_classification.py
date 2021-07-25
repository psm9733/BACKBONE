import os
import torch
import albumentations
import albumentations.pytorch
import warnings
from datetime import datetime
from torch.utils.data import DataLoader
from utils.generator import TinyImageNet
from network.model import Classification
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    model_name="DenseNext32"
    input_shape = (3, 64, 64)
    num_classes = 200
    batch_size = 32
    worker = 4
    max_lr = 1e-3
    min_lr = 1e-4
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
    tb_logger = pl_loggers.TensorBoardLogger(logdir, name = model_name, default_hp_metric=False)
    engine = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if engine == torch.device('cpu'):
        warnings.warn('Cannot use CUDA context. Train might be slower!')
    model = Classification(classes=num_classes, min_lr=min_lr, max_lr=max_lr, weight_decay=weight_decay)

    # data setup
    train_transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Affine(),
        albumentations.ColorJitter(),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])
    valid_transform = albumentations.Compose([
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])

    trainLoader = DataLoader(
        TinyImageNet('/home/fssv1/sangmin/backbone/dataset/tiny-imagenet-200', True, False, train_transform, num_classes),
        batch_size=batch_size, shuffle=True, num_workers=worker, drop_last=True)
    validLoader = DataLoader(
        TinyImageNet('/home/fssv1/sangmin/backbone/dataset/tiny-imagenet-200', False, False, valid_transform, num_classes),
        batch_size=batch_size, num_workers=worker)
    pl.seed_everything(1)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename=model_name, monitor="val_top1", mode='min', verbose=True, save_top_k=1)
    trainer = pl.Trainer(precision=16, max_epochs=max_epochs, gpus=1, accumulate_grad_batches = 1, logger=tb_logger, callbacks=checkpoint_callback)
    trainer.fit(model, trainLoader, validLoader)

if __name__ == "__main__":
    main()
