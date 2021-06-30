import tqdm
import torch
import albumentations
import albumentations.pytorch
import warnings
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import Classification
from utils.utils import weight_initialize
from torchsummary import summary

if __name__ == "__main__":
    activation = nn.ReLU()
    input_shape = (3, 224, 224)
    classes = 200
    model = Classification(activation, classes)
    weight_initialize(model)
    # summary(model, input_shape, batch_size=4, device='cpu')

    ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if ctx == torch.device('cpu'):
        warnings.warn(
            'Cannot use CUDA context. Train might be slower!')

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

    # trainLoader = DataLoader(dataset.TinyImageNet(
    #     'E:/FSNet2/Datasets/tiny-imagenet-200', True, False, transform), batch_size=args.batch_size,
    #     shuffle=True, num_workers=args.workers, drop_last=True)
    # validLoader = DataLoader(dataset.TinyImageNet(
    #     'E:/FSNet2/Datasets/tiny-imagenet-200', False, False, valid_transform), batch_size=args.batch_size,
    #     num_workers=args.workers)

    # # training setup
    # optimizer = torch.optim.SGD(
    #     model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, args.lr / 10, args.lr, mode='triangular', step_size_up=trainLoader.__len__() * 4)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # logger = Logger(args.outdir, args.log_freq, args.save_freq)
    # metric = TopKAccuracy(one_hot=False)

    # # Fit
    # logdata = dict()
    # max_epoch = make_divisible(args.epochs, 4)
    # print('max epoch: ', max_epoch)
    # for epochs in range(max_epoch):
    #     model.train()
    #     metric.clear()
    #     iterator = tqdm.tqdm(enumerate(trainLoader),
    #                          total=trainLoader.__len__(), desc='')
    #     for batch, sample in iterator:
    #         x = sample['img'].to(ctx)
    #         y_true = sample['y_true'].to(ctx)
    #         y_pred = model(x)['pred']
    #         loss = loss_fn(y_pred, y_true)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         metric.step(y_pred, y_true)
    #         logdata['train_loss'] = loss.item()
    #         for k, v in metric.accuracy.items():
    #             logdata['train_top{}'.format(k)] = v
    #         logdata['lr'] = optimizer.param_groups[0]['lr']
    #         logger.step(logdata, model)
    #         iterator.set_description("epoch: {0}, iter: {1}/{2}, loss: {3:0.4f}, train acc:{4:0.4f}".format(
    #             epochs, batch, trainLoader.__len__(), logdata['train_loss'], logdata['train_top1']))
    #
    #     if epochs % args.val_freq == args.val_freq - 1:
    #         model.eval()
    #         metric.clear()
    #         with torch.no_grad():
    #             iterator = tqdm.tqdm(enumerate(validLoader),
    #                                  total=validLoader.__len__())
    #             for batch, sample in iterator:
    #                 x = sample['img'].to(ctx)
    #                 y_true = sample['y_true'].to(ctx)
    #                 y_pred = model(x)['pred']
    #                 loss = loss_fn(y_pred, y_true)
    #                 logdata['valid_loss'] = loss.item()
    #                 metric.step(y_pred, y_true)
    #                 iterator.set_description("epoch: {0}, iter: {1}/{2}, loss: {3:0.4f}".format(
    #                     epochs, batch, validLoader.__len__(), logdata['valid_loss']))
    #             for k, v in metric.accuracy.items():
    #                 logdata['valid_top{}'.format(k)] = v


