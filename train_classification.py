import tqdm
import torch
import albumentations
import albumentations.pytorch
import warnings
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import Classification
from utils.utils import weight_initialize
from utils.generator import TinyImageNet
from utils.logger import Logger
from utils.metric import TopKAccuracy
from utils.utils import make_divisible
from model.model import Classification
from torchsummary import summary


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
    save_freq = 5000
    logdir = "./logs"
    max_epoch = 1000
    model = Classification(activation, num_classes)
    summary(model, input_shape, batch_size=batch_size, device='cpu')
    weight_initialize(model)
    ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if ctx == torch.device('cpu'):
        warnings.warn('Cannot use CUDA context. Train might be slower!')
    model.to(ctx)

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

    logger = Logger(logdir, log_freq, save_freq)
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
            x = sample['img'].to(ctx)
            y_true = sample['y_true'].to(ctx)
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
            logger.step(logdata, model)
            iterator.set_description("epoch: {0}, iter: {1}/{2}, loss: {3:0.4f}, train acc:{4:0.4f}".format(
                epochs, batch, trainLoader.__len__(), logdata['train_loss'], logdata['train_top1']))

        if epochs % val_freq == val_freq - 1:
            model.eval()
            metric.clear()
            with torch.no_grad():
                iterator = tqdm.tqdm(enumerate(validLoader),
                                     total=validLoader.__len__())
                for batch, sample in iterator:
                    x = sample['img'].to(ctx)
                    y_true = sample['y_true'].to(ctx)
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
