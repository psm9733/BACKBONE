import os
import tqdm
import torch
import albumentations
import albumentations.pytorch
import warnings
import torch.nn as nn
import utils.pytorch_ssim as pytorch_ssim
import cv2
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from utils.utils import weight_initialize
from utils.generator import NoiseReduction
from utils.logger import Logger
from utils.saver import Saver
from utils.utils import make_divisible
from model.model import Segmentation
from torchsummary import summary
from adamp import *
import torch.nn.functional as F

def main():
    activation = nn.ReLU()
    input_shape = (3, 384, 512)
    batch_size = 8
    feature_num = 512

    worker = 1
    learning_rate = 1e-2
    weight_decay = 1e-4
    log_freq = 100
    val_freq = 5
    save_freq = 100
    max_epoch = 2000
    opt_level = 'O1'
    timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    logdir = "./logs/" + timestamp
    save_dir = "./saved_model/" + timestamp
    if os.path.isdir('./log_img') == False:
        os.mkdir('./log_img')
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
    # optimizer = AdamP(model.parameters(), learning_rate)
    optimizer = SGDP(model.parameters(), learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, learning_rate / 10, learning_rate, mode='triangular', step_size_up=trainLoader.__len__() * 4)
    loss_fn = torch.nn.MSELoss()
    # loss_fn = pytorch_ssim.SSIM(window_size = 11)
    logger = Logger(logdir, log_freq)
    logger_writer = logger.getSummaryWriter()
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
                for index in range(batch_size):
                    # logger_writer.add_image('input img', x[index], logger.getStep())
                    # logger_writer.add_image('gt img', y_true[index], logger.getStep())
                    if logger.getStep() % 100 == 0:
                        input_img = torch.clone(x[index])
                        input_img = torch.squeeze(input_img.to("cpu"))
                        input_img = torch.permute(input_img, (2, 1, 0))
                        input_img = (input_img.detach().numpy() * 255).astype(np.uint8)
                        input_img = cv2.resize(input_img, (416, 416))

                        gt_img = torch.clone(y_true[index])
                        gt_img = torch.squeeze(gt_img.to("cpu"))
                        gt_img = torch.permute(gt_img, (2, 1, 0))
                        gt_img = (gt_img.detach().numpy() * 255).astype(np.uint8)
                        gt_img = cv2.resize(gt_img, (416, 416))

                        pred_img = torch.clone(y_pred[index])
                        pred_img = torch.squeeze(pred_img.to("cpu"))
                        pred_img = torch.permute(pred_img, (2, 1, 0))
                        pred_img = (pred_img.detach().numpy() * 255).astype(np.uint8)
                        pred_img = cv2.resize(pred_img, (416, 416))

                        cv2.imwrite("./log_img/input_img_{}.jpg".format(logger.getStep()), input_img)
                        cv2.imwrite("./log_img/gt_img_{}.jpg".format(logger.getStep()), gt_img)
                        cv2.imwrite("./log_img/pred_img_{}.jpg".format(logger.getStep()), pred_img)
                    # logger_writer.add_image('predict img', pred_img, logger.getStep())
                # for index in range(0, batch_size):
                    # pre_img = torch.clone(y_pred[0])
                    # pre_img = torch.squeeze(pre_img.to("cpu"))
                    # pre_img = torch.permute(pre_img, (1, 2, 0))
                    # pre_img = pre_img.detach().numpy()
                    # pre_img = pre_img * 255
                    # pre_img = pre_img.astype(np.uint8)
                    # grid = torchvision.utils.make_grid([x[index], y_true[index], y_pred[index]])
                    # logger_writer.add_image('img', grid, logger.getStep())
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