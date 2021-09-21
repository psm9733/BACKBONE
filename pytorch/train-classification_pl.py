import os
import torch
import albumentations
import albumentations.pytorch
import warnings
from datetime import datetime
from utils.generator import Mnist, TinyImageNet
from network.model import Classification
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    model_name="DenseNext20"
    batch_size = 32
    weight_decay = 1e-4
    max_epochs = 1000
    workers = 4
    timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    logdir = "/Users/sangmin/BACKBONE/pytorch/logs/" + timestamp
    save_dir = "/Users/sangmin/BACKBONE/pytorch/saved_model/" + timestamp
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

    # data setup
    train_transform = albumentations.Compose([
        albumentations.SomeOf([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(-90, 90),
            albumentations.Sharpen(),
        ], 2, p=0.5),
        albumentations.Affine(),
        albumentations.ColorJitter(),
        # albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])
    valid_transform = albumentations.Compose([
        # albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])
    checkpoint_callback = ModelCheckpoint(monitor = "val_loss",
                                   verbose=True,
                                   dirpath=save_dir,
                                   filename="{epoch}_{val_loss:.4f}",
                                   save_top_k = 1)
    trainer = pl.Trainer(auto_lr_find=False, precision=32, max_epochs=max_epochs, gpus=0, accumulate_grad_batches = 1, logger=tb_logger, callbacks=checkpoint_callback)
    model = Classification(task="color_mnist", batch_size=batch_size, train_aug=train_transform, val_aug=valid_transform, workers=workers, weight_decay=weight_decay)
    # trainer.tune(model)
    trainer.fit(model)

if __name__ == "__main__":
    main()
