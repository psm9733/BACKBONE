import os
import tqdm
import torch
import albumentations
import albumentations.pytorch
import warnings
import torch.nn as nn
import utils.pytorch_ssim as pytorch_ssim
from datetime import datetime
from torch.utils.data import DataLoader
from utils.utils import weight_initialize
from utils.generator import NoiseReduction
from utils.logger import Logger
from utils.saver import Saver
from utils.utils import make_divisible
from model.model import Segmentation
from torchsummary import summary

def main():
    activation = nn.ReLU()
    input_shape = (3, 2448, 3264)
    batch_size = 2
    feature_num = 32

    worker = 4
    learning_rate = 1e-3
    weight_decay = 1e-4
    log_freq = 100
    val_freq = 5
    save_freq = 1000
    max_epoch = 100
    opt_level = 'O1'
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

    model = Segmentation(activation, feature_num)
    summary(model, input_shape, batch_size=batch_size, device='cpu')
    weight_initialize(model)
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
            albumentations.RandomResizedCrop(height=input_shape[1], width=input_shape[2]),
            # albumentations.ColorJitter(),
        ], 3, p=0.5),
        # albumentations.OneOf([
        #     albumentations.MotionBlur(p=1),
        #     albumentations.OpticalDistortion(p=1),
        #     albumentations.GaussNoise(p=1)
        # ], p=0.5),
        albumentations.pytorch.ToTensorV2(),
    ], additional_targets={'image1': 'image', 'image2': 'image'})
    valid_transform = albumentations.Compose([
        albumentations.Resize(height=input_shape[1], width=input_shape[2]),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], additional_targets={'image1': 'image', 'image2': 'image'})

    trainLoader = DataLoader(NoiseReduction('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', True, input_shape, train_transform), batch_size=batch_size, shuffle=True, num_workers=worker, drop_last=True)
    validLoader = DataLoader(NoiseReduction('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', False, input_shape, valid_transform), batch_size=batch_size, num_workers=worker)

    # training setup
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, learning_rate / 10, learning_rate, mode='triangular', step_size_up=trainLoader.__len__() * 4)
    # loss_fn = torch.nn.Loss()
    loss_fn = pytorch_ssim.SSIM(window_size = 11)
    logger = Logger(logdir, log_freq)
    saver = Saver(save_dir, save_freq)

    # Fit
    logdata = dict()
    max_epoch = make_divisible(max_epoch, 4)
    print('max epoch: ', max_epoch)
    for epochs in range(max_epoch):
        model.train()
        iterator = tqdm.tqdm(enumerate(trainLoader), total=trainLoader.__len__(), desc='')
        for batch, sample in iterator:
            with torch.cuda.amp.autocast():
                x = sample['img'].to(device)
                y_true = sample['y_true'].to(device)
                y_pred = model(x)['pred']
                loss = loss_fn(y_pred, y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                logdata['train_loss'] = loss.item()
                logdata['lr'] = optimizer.param_groups[0]['lr']
                logger.step(logdata)
                saver.step(model)
                iterator.set_description("epoch: {0}, iter: {1}/{2}, loss: {3:0.4f}".format(
                    epochs, batch, trainLoader.__len__(), logdata['train_loss']))

        if epochs % val_freq == val_freq - 1:
            model.eval()
            with torch.no_grad():
                iterator = tqdm.tqdm(enumerate(validLoader),
                                     total=validLoader.__len__())
                for batch, sample in iterator:
                    x = sample['img'].to(device)
                    y_true = sample['y_true'].to(device)
                    y_pred = model(x)['pred']
                    loss = loss_fn(y_pred, y_true)
                    logdata['valid_loss'] = loss.item()
                    iterator.set_description("epoch: {0}, iter: {1}/{2}, loss: {3:0.4f}".format(
                        epochs, batch, trainLoader.__len__(), logdata['train_loss']))

if __name__ == "__main__":
    main()