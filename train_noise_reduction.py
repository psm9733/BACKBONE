import tqdm
import torch
import albumentations
import albumentations.pytorch
import warnings
import torch.nn as nn
import datetime as pydatetime
import os
from torch.utils.data import DataLoader
from model.model import Classification
from utils.utils import weight_initialize
from utils.generator import NoiseReduction
from utils.logger import Logger
from utils.metric import TopKAccuracy
from utils.utils import make_divisible
from model.model import Segmentation
from torchsummary import summary

def main():
    activation = nn.ReLU()
    input_shape = (3, 320, 320)
    batch_size = 2
    worker = 4
    learning_rate = 1e-3
    weight_decay = 1e-4
    log_freq = 100
    val_freq = 5
    save_freq = 5000
    logdir = "./logs/" + str(pydatetime.datetime.now().timestamp())
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)

    max_epoch = 100
    model = Segmentation(activation)
    summary(model, input_shape, batch_size=batch_size, device='cpu')
    weight_initialize(model)
    ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if ctx == torch.device('cpu'):
        warnings.warn('Cannot use CUDA context. Train might be slower!')
    model.to(ctx)

    # data setup
    train_transform = albumentations.Compose([
        albumentations.Resize(height=input_shape[1], width=input_shape[2]),
        albumentations.Normalize(0, 1),
        albumentations.SomeOf([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(-90, 90),
            albumentations.RandomResizedCrop(height=input_shape[1], width=input_shape[2]),
            # albumentations.ColorJitter(),
        ], 3, p=0.5),
        # albumentations.OneOf([
        #     albumentations.MotionBlur(p=1),
        #     albumentations.OpticalDistortion(p=1),
        #     albumentations.GaussNoise(p=1)
        # ], p=0.5),
        albumentations.pytorch.ToTensorV2(),
    ])
    valid_transform = albumentations.Compose([
        albumentations.Resize(height=input_shape[1], width=input_shape[2]),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])

    trainLoader = DataLoader(NoiseReduction('C:/Users/sangmin/Desktop/backbone/dataset/lg_noise_remove', True, input_shape, train_transform), batch_size=batch_size, shuffle=True, num_workers=worker, drop_last=True)
    validLoader = DataLoader(NoiseReduction('C:/Users/sangmin/Desktop/backbone/dataset/lg_noise_remove', False, input_shape, valid_transform), batch_size=batch_size, num_workers=worker)

    # training setup
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, learning_rate / 10, learning_rate, mode='triangular', step_size_up=trainLoader.__len__() * 4)
    loss_fn = torch.nn.MSELoss()
    logger = Logger(logdir, log_freq, save_freq)

    # Fit
    logdata = dict()
    max_epoch = make_divisible(max_epoch, 4)
    print('max epoch: ', max_epoch)
    for epochs in range(max_epoch):
        model.train()
        iterator = tqdm.tqdm(enumerate(trainLoader),
                             total=trainLoader.__len__(), desc='')
        for batch, sample in iterator:
            x = sample['img'].to(ctx)
            y_true = sample['y_true'].to(ctx)
            y_pred = model(x)['pred']
            loss = loss_fn(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            logdata['train_loss'] = loss.item()
            logdata['lr'] = optimizer.param_groups[0]['lr']
            logger.step(logdata, model)
            iterator.set_description("epoch: {0}, iter: {1}/{2}, loss: {3:0.4f}".format(
                epochs, batch, trainLoader.__len__(), logdata['train_loss']))


        if epochs % val_freq == val_freq - 1:
            model.eval()
            with torch.no_grad():
                iterator = tqdm.tqdm(enumerate(validLoader),
                                     total=validLoader.__len__())
                for batch, sample in iterator:
                    x = sample['img'].to(ctx)
                    y_true = sample['y_true'].to(ctx)
                    y_pred = model(x)['pred']
                    loss = loss_fn(y_pred, y_true)
                    logdata['valid_loss'] = loss.item()
                    iterator.set_description("epoch: {0}, iter: {1}/{2}, loss: {3:0.4f}".format(
                        epochs, batch, trainLoader.__len__(), logdata['train_loss']))


if __name__ == "__main__":
    main()