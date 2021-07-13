import os
import tqdm
import torch
import albumentations
import albumentations.pytorch
import warnings
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from utils.utils import weight_initialize
from utils.generator import TinyImageNet
from utils.logger import Logger
from utils.saver import Saver
from utils.metric import TopKAccuracy
from utils.utils import make_divisible
from network.model import Classification

def main():
    activation = nn.ReLU()
    input_shape = (3, 64, 64)
    num_classes = 200
    batch_size = 16
    worker = 4
    learning_rate = 1e-3
    weight_decay = 1e-4
    log_freq = 100
    val_freq = 5
    save_freq = 10000
    max_epoch = 200
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

    model = Classification(activation, num_classes)
    summary(model, input_shape, batch_size=batch_size, device='cpu')
    weight_initialize(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        warnings.warn('Cannot use CUDA context. Train might be slower!')
    model.to(device)

    # data setup
    train_transform = albumentations.Compose([
        # albumentations.HorizontalFlip(p=0.5),
        # albumentations.VerticalFlip(p=0.5),
        # albumentations.Affine(),
        # albumentations.ColorJitter(),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])
    valid_transform = albumentations.Compose([
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])

    trainLoader = DataLoader(
        TinyImageNet('C:/Users/sangmin/Desktop/backbone/dataset/tiny-imagenet-200', True, False, train_transform, num_classes),
        batch_size=batch_size, shuffle=True, num_workers=worker, drop_last=True)
    validLoader = DataLoader(
        TinyImageNet('C:/Users/sangmin/Desktop/backbone/dataset/tiny-imagenet-200', False, False, valid_transform, num_classes),
        batch_size=batch_size, num_workers=worker)

    # training setup
    optimizer = torch.optim.SGD(
        model.parameters(), learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, learning_rate / 10, learning_rate, mode='triangular', step_size_up=trainLoader.__len__() * 4)
    loss_fn = torch.nn.CrossEntropyLoss()

    logger = Logger(logdir, log_freq)
    saver = Saver(save_dir, save_freq)
    metric = TopKAccuracy(one_hot=False)

    # Fit
    logdata = dict()
    # max_epoch = make_divisible(max_epoch, 4)
    print('max epoch: ', max_epoch)
    for epochs in range(max_epoch):
        model.train()
        metric.clear()
        iterator = tqdm.tqdm(enumerate(trainLoader),
                             total=trainLoader.__len__(), desc='')
        for batch, sample in iterator:
            x = sample['img'].to(device)
            y_true = sample['y_true'].to(device)
            y_pred = model(x)['pred']
            loss = loss_fn(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            metric.step(y_pred, y_true)
            logdata['train_loss'] = loss.item()
            for k, v in metric.accuracy.items():
                logdata['train_top{}'.format(k)] = v
            logdata['lr'] = optimizer.param_groups[0]['lr']
            logger.step(logdata)
            saver.step(model)
            iterator.set_description("epoch: {0}, iter: {1}/{2}, loss: {3:0.4f}, train acc:{4:0.4f}".format(
                epochs, batch, trainLoader.__len__(), logdata['train_loss'], logdata['train_top1']))

        if epochs % val_freq == val_freq - 1:
            model.eval()
            metric.clear()
            with torch.no_grad():
                iterator = tqdm.tqdm(enumerate(validLoader),
                                     total=validLoader.__len__())
                for batch, sample in iterator:
                    x = sample['img'].to(device)
                    y_true = sample['y_true'].to(device)
                    y_pred = model(x)['pred']
                    loss = loss_fn(y_pred, y_true)
                    logdata['valid_loss'] = loss.item()
                    metric.step(y_pred, y_true)
                    iterator.set_description("epoch: {0}, iter: {1}/{2}, loss: {3:0.4f}".format(
                        epochs, batch, validLoader.__len__(), logdata['valid_loss']))
                for k, v in metric.accuracy.items():
                    logdata['valid_top{}'.format(k)] = v

if __name__ == "__main__":
    main()
