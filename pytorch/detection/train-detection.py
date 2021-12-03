import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import albumentations
import albumentations.pytorch
import warnings
from datetime import datetime
from network.model import Scaled_Yolov4
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import get_anchors
from config import *
torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    backbone_name = "Regnet-yolo"
    input_shape = (3, 608, 608)
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 5e-4
    classes = 80
    model_name = backbone_name + "/lr=" + str(learning_rate) + "/wd=" + str(weight_decay) + "/batchsize=" + str(batch_size)
    max_epochs = 125
    workers = 1
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
    engine = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if engine == torch.device('cpu'):
        warnings.warn('Cannot use CUDA context. Train might be slower!')

    # data setup
    train_transform = albumentations.Compose([
        albumentations.Resize(height=input_shape[1], width=input_shape[2]),
        albumentations.SomeOf([
            albumentations.Flip(),
            albumentations.Sharpen(),
        ], 2, p=0.5),
        albumentations.Affine(),
        albumentations.ColorJitter(),
        # albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))
    valid_transform = albumentations.Compose([
        # albumentations.Normalize(0, 1),
        albumentations.Resize(height=input_shape[1], width=input_shape[2]),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))
    checkpoint_callback = ModelCheckpoint(monitor = "val_loss",
                                   verbose=True,
                                   dirpath=save_dir,
                                   filename="{epoch}_{val_loss:.4f}",
                                   save_top_k = 1)
    trainer = pl.Trainer(auto_lr_find=False, precision=32, max_epochs=max_epochs, gpus=1, accumulate_grad_batches = 1, logger=tb_logger, callbacks=checkpoint_callback)
    model = Scaled_Yolov4(input_shape, classes=classes, batch_size=batch_size, train_aug=train_transform, val_aug=valid_transform, workers=workers, learning_rate=learning_rate, weight_decay=weight_decay)
    model.set_train_path(TRAIN_DIR_PATH)
    model.set_valid_path(VALID_DIR_PATH)
    trainer.fit(model)

if __name__ == "__main__":
    main()
    # anchors = get_anchors("/home/fssv1/sangmin/backbone/dataset/coco/train2017/")