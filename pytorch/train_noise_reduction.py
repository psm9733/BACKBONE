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
from network.model import DeNoising
from utils.losses import PSNRLoss
from torchsummary import summary
from adamp import *
import os

import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    activation = nn.PReLU()
    input_shape = (3, 256, 256)
    batch_size = 10
    feature_num = 256

    worker = 1
    max_lr = 1e-4
    min_lr = 1e-7
    weight_decay = 1e-5
    log_freq = 100
    val_freq = 5
    save_freq = 100
    max_epoch = 2000
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

    model = DeNoising(activation, feature_num, groups=32)
    # model = nn.DataParallel(model).to("cuda")
    # summary(model, input_shape, batch_size=batch_size, device='cpu')
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
            # albumentations.RandomResizedCrop(height=input_shape[1], width=input_shape[2]),
        #     # albumentations.ColorJitter(),
        ], 2, p=0.5),
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
    # validLoader = DataLoader(NoiseReduction('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', False, input_shape, valid_transform), batch_size=batch_size, num_workers=worker)

    # training setup
    optimizer = AdamP(model.parameters(), min_lr, weight_decay = weight_decay)
    # optimizer = SGDP(model.parameters(), learning_rate, momentum=0.9, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, learning_rate / 10, learning_rate, mode='triangular', step_size_up=trainLoader.__len__() * 4)
    # loss_ssim = pytorch_ssim.SSIM(window_size = 11)
    loss_psnr = PSNRLoss(max_val=1)
    # loss_mse = torch.nn.MSELoss()
    logger = Logger(logdir, log_freq)
    logger_writer = logger.getSummaryWriter()
    saver = Saver(save_dir, save_freq)

    # Fit
    logdata = dict()
    # max_epoch = make_divisible(max_epoch, 4)
    print('max epoch: ', max_epoch)
    iterator = tqdm.tqdm(enumerate(trainLoader), total=trainLoader.__len__(), desc='')
    step_length = len(iterator)
    for epochs in range(max_epoch):
        model.train()
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=int(step_length / 2), cycle_momentum = False, mode='triangular2', gamma=0.995, verbose=True)
        for batch, sample in iterator:
            with torch.cuda.amp.autocast():
                x = sample['img'].to(device)
                y_true = sample['y_true'].to(device)
                y_pred_hg1 = model(x)['hg1_loss']
                y_pred_hg2 = model(x)['hg2_loss']
                # y_pred_hg3 = model(x)['hg3_loss']
                # y_pred_hg4 = model(x)['hg4_loss']
                for index in range(batch_size):
                    # logger_writer.add_image('input img', x[index], logger.getStep())
                    # logger_writer.add_image('gt img', y_true[index], logger.getStep())
                    if logger.getStep() % 50 == 0:
                        input_img = torch.clone(x[index])
                        input_img = torch.squeeze(input_img.to("cpu"))
                        input_img = torch.permute(input_img, (2, 1, 0))
                        input_img = (input_img.detach().numpy() * 255).astype(np.uint8)
                        input_img = cv2.resize(input_img, (input_shape[1], input_shape[2]))

                        gt_img = torch.clone(y_true[index])
                        gt_img = torch.squeeze(gt_img.to("cpu"))
                        gt_img = torch.permute(gt_img, (2, 1, 0))
                        gt_img = (gt_img.detach().numpy() * 255).astype(np.uint8)
                        gt_img = cv2.resize(gt_img, (input_shape[1], input_shape[2]))

                        pred_hg1_img = torch.clone(y_pred_hg1[index])
                        pred_hg1_img = torch.clip(pred_hg1_img, 0, 1)
                        pred_hg1_img = torch.squeeze(pred_hg1_img.to("cpu"))
                        pred_hg1_img = torch.permute(pred_hg1_img, (2, 1, 0))
                        pred_hg1_img = (pred_hg1_img.detach().numpy() * 255).astype(np.uint8)
                        pred_hg1_img = cv2.resize(pred_hg1_img, (input_shape[1], input_shape[2]))

                        pred_hg2_img = torch.clone(y_pred_hg2[index])
                        pred_hg2_img = torch.clip(pred_hg2_img, 0, 1)
                        pred_hg2_img = torch.squeeze(pred_hg2_img.to("cpu"))
                        pred_hg2_img = torch.permute(pred_hg2_img, (2, 1, 0))
                        pred_hg2_img = (pred_hg2_img.detach().numpy() * 255).astype(np.uint8)
                        pred_hg2_img = cv2.resize(pred_hg2_img, (input_shape[1], input_shape[2]))

                        # pred_hg3_img = torch.clone(y_pred_hg3[index])
                        # pred_hg3_img = torch.clip(pred_hg3_img, 0, 1)
                        # pred_hg3_img = torch.squeeze(pred_hg3_img.to("cpu"))
                        # pred_hg3_img = torch.permute(pred_hg3_img, (2, 1, 0))
                        # pred_hg3_img = (pred_hg3_img.detach().numpy() * 255).astype(np.uint8)
                        # pred_hg3_img = cv2.resize(pred_hg3_img, (input_shape[1], input_shape[2]))
                        #
                        # pred_hg4_img = torch.clone(y_pred_hg4[index])
                        # pred_hg4_img = torch.clip(pred_hg4_img, 0, 1)
                        # pred_hg4_img = torch.squeeze(pred_hg4_img.to("cpu"))
                        # pred_hg4_img = torch.permute(pred_hg4_img, (2, 1, 0))
                        # pred_hg4_img = (pred_hg4_img.detach().numpy() * 255).astype(np.uint8)
                        # pred_hg4_img = cv2.resize(pred_hg4_img, (input_shape[1], input_shape[2]))

                        cv2.imwrite("./log_img/{}_input_img.jpg".format(logger.getStep()), input_img)
                        cv2.imwrite("./log_img/{}_gt_img.jpg".format(logger.getStep()), gt_img)
                        cv2.imwrite("./log_img/{}_pred_hg1_img.jpg".format(logger.getStep()), pred_hg1_img)
                        cv2.imwrite("./log_img/{}_pred_hg2_img.jpg".format(logger.getStep()), pred_hg2_img)
                        # cv2.imwrite("./log_img/{}_pred_hg3_img.jpg".format(logger.getStep()), pred_hg3_img)
                        # cv2.imwrite("./log_img/{}_pred_hg4_img.jpg".format(logger.getStep()), pred_hg4_img)
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
                loss = loss_psnr(y_pred_hg1, y_true)
                loss += loss_psnr(y_pred_hg2, y_true)
                # loss += loss_psnr(y_pred_hg3, y_true)
                # loss += loss_psnr(y_pred_hg4, y_true)
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

        # if epochs % val_freq == val_freq - 1:
        #     model.eval()
        #     with torch.no_grad():
        #         iterator = tqdm.tqdm(enumerate(validLoader),
        #                              total=validLoader.__len__())
        #         for batch, sample in iterator:
        #             x = sample['img'].to(device)
        #             y_true = sample['y_true'].to(device)
        #             y_pred_hg1 = model(x)['hg1_loss']
        #             y_pred_hg2 = model(x)['hg2_loss']
        #             y_pred_hg3 = model(x)['hg3_loss']
        #             y_pred_hg4 = model(x)['hg4_loss']
        #             loss = loss_psnr(y_pred_hg1, y_true)
        #             loss += loss_psnr(y_pred_hg2, y_true)
        #             loss += loss_psnr(y_pred_hg3, y_true)
        #             loss += loss_psnr(y_pred_hg4, y_true)
        #             logdata['valid_loss'] = loss.item()
        #             iterator.set_description("epoch: {0}, iter: {1}/{2}, loss: {3:0.4f}".format(
        #                 epochs, batch, trainLoader.__len__(), logdata['train_loss']))

if __name__ == "__main__":
    main()