import torch
import albumentations
import albumentations.pytorch
import warnings
from datetime import datetime
from network.model import DeNoising
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import os

import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    backbone_name="SHNet"
    input_shape = (3, 256, 256)
    feature_num = 512
    batch_size = 8
    workers = 4
    learning_rate = 1e-7
    weight_decay = 1e-5
    model_name = backbone_name + "/lr=" + str(learning_rate) + "/wd=" + str(weight_decay) + "/batchsize=" + str(batch_size)
    max_epochs = 72
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        warnings.warn('Cannot use CUDA context. Train might be slower!')
    else:
        warnings.warn('Single GPU Activate!')

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
    model = DeNoising(feature_num=feature_num, input_shape=input_shape, batch_size=batch_size, train_aug=train_transform, val_aug=valid_transform, workers=workers, learning_rate=learning_rate, weight_decay=weight_decay)
    # model.to(device)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename=model_name, monitor="PSNR_loss", mode='min', verbose=True, save_top_k=1)
    trainer = pl.Trainer(auto_lr_find=False, precision=16, max_epochs=max_epochs, gpus=1, accumulate_grad_batches = 1, logger=tb_logger, callbacks=checkpoint_callback)
    # trainer.tune(model)
    trainer.fit(model)

if __name__ == "__main__":
    main()