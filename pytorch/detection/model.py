from network.regnet.regnet import *
from network.common.blocks import *
from network.common.neck import *
from network.common.head import *
from utils.utils import weight_initialize, read_anchors
from torch.utils.data import DataLoader
from adamp import *
from detection.generator import *
from detection.losses import YoloLoss
import pytorch_lightning as pl
import torch.nn
import torch


class Scaled_Yolov4(pl.LightningModule):
    def __init__(self, input_shape, classes, batch_size=1, train_aug=None, val_aug=None, workers=1, learning_rate=1e-4,
                 weight_decay=1e-4):
        super().__init__()
        # hyper parameter
        self.input_shape = input_shape
        self.classes = classes
        self.batch_size = batch_size
        self.in_channels = self.input_shape[0]
        self.lr = learning_rate
        self.train_aug = train_aug
        self.val_aug = val_aug
        self.workers = workers
        self.weight_decay = weight_decay
        self.train_dir = None
        self.valid_dir = None

        # model setting
        self.activation = nn.ReLU()
        self.anchor = read_anchors(ANCHOR_INFO_PATH)
        self.branch_num = 3
        self.anchor_num = int(len(self.anchor) / self.branch_num)
        self.stem_in_channels_list = [self.in_channels, 32, 64]
        self.stem_out_channels_list = [32, 64, 128]
        self.fpn_out_channels_list = [128, 256, 512]
        self.head_out_channels_list = [int(self.anchor_num * (self.classes + 5)),
                                       int(self.anchor_num * (self.classes + 5)),
                                       int(self.anchor_num * (self.classes + 5))]
        self.stem = StemBlock_3(self.stem_in_channels_list, self.stem_out_channels_list, self.activation, bias=True)
        self.backbone = RegNetX_mini_3stage(self.activation, self.stem.getOutputChannels(), bias=True)
        self.fpn_neck = FPN_3branch(self.backbone.getOutputBranchChannels()[
                                    len(self.backbone.getOutputBranchChannels()) - self.branch_num:self.branch_num + 1],
                                    self.fpn_out_channels_list, self.activation, bias=True)
        self.yolo_head = Yolo_3branch(self.fpn_neck.getOutputBranchChannels(), self.head_out_channels_list, bias=True)
        self.output_shape = [
            [self.head_out_channels_list[0],
             int(input_shape[1] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 4),
             int(input_shape[2] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 4)],
            [self.head_out_channels_list[1],
             int(input_shape[1] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 2),
             int(input_shape[2] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 2)],
            [self.head_out_channels_list[2],
             int(input_shape[1] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 1),
             int(input_shape[2] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 1)],
        ]
        weight_initialize(self.stem)
        weight_initialize(self.backbone)
        weight_initialize(self.fpn_neck)
        weight_initialize(self.yolo_head)

    def set_train_path(self, dir_path):
        self.train_dir = dir_path

    def set_valid_path(self, dir_path):
        self.valid_dir = dir_path

    def setup(self, stage):
        self.train_gen = YoloGenerator(self.train_dir, self.input_shape, self.output_shape, self.classes, self.anchor,
                                       True, self.train_aug)
        self.val_gen = YoloGenerator(self.valid_dir, self.input_shape, self.output_shape, self.classes, self.anchor,
                                     False, self.val_aug)

    def forward(self, input):
        stem_out = self.stem(input)
        big_out, middle_out, small_out = self.backbone(stem_out)[
                                         len(self.backbone.getOutputBranchChannels()) - self.branch_num:self.branch_num + 1]
        big_out, middle_out, small_out = self.fpn_neck(big_out, middle_out, small_out)
        big_out, middle_out, small_out = self.yolo_head(big_out, middle_out, small_out)
        return {"big_out": big_out, "middle_out": middle_out, "small_out": small_out}

    def configure_optimizers(self):
        self.optimizer = AdamP(self.parameters(), self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=self.lr * 10,
                                                           step_size_up=20, cycle_momentum=False, mode='triangular2',
                                                           gamma=0.995, verbose=True)
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True, drop_last=True,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_gen, batch_size=max(1, int(self.batch_size / 4)), shuffle=False, drop_last=True,
                          num_workers=self.workers)

    def training_step(self, train_batch, batch_idx):
        x = train_batch['img']
        gt_big_out = train_batch['big_out']
        gt_middle_out = train_batch['middle_out']
        gt_small_out = train_batch['small_out']
        y_true = [gt_big_out, gt_middle_out, gt_small_out]
        gt_big_out_shape = gt_big_out.shape
        gt_middle_out_shape = gt_middle_out.shape
        gt_small_out_shape = gt_small_out.shape
        pred_big_out = self.forward(x)["big_out"]
        pred_middle_out = self.forward(x)["middle_out"]
        pred_small_out = self.forward(x)["small_out"]
        pred_big_out = pred_big_out.reshape(gt_big_out_shape[0], gt_big_out_shape[1], gt_big_out_shape[2],
                                            gt_big_out_shape[3], gt_big_out_shape[4])
        pred_middle_out = pred_middle_out.reshape(gt_middle_out_shape[0], gt_middle_out_shape[1],
                                                  gt_middle_out_shape[2], gt_middle_out_shape[3],
                                                  gt_middle_out_shape[4])
        pred_small_out = pred_small_out.reshape(gt_small_out_shape[0], gt_small_out_shape[1], gt_small_out_shape[2],
                                                gt_small_out_shape[3], gt_small_out_shape[4])
        y_pred = [pred_big_out, pred_middle_out, pred_small_out]
        loss_fn = YoloLoss(self.input_shape, self.classes, self.branch_num, self.anchor, self.batch_size)
        total_loss, confidence_loss, location_loss, class_loss = loss_fn(y_pred, y_true)
        self.log("lr", self.scheduler.get_lr()[0])
        self.log("confidence_loss", confidence_loss, on_epoch=True, prog_bar=True)
        self.log("location_loss", location_loss, on_epoch=True, prog_bar=True)
        self.log("class_loss", class_loss, on_epoch=True, prog_bar=True)
        self.log("total_loss", total_loss, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch['img']
        gt_big_out = val_batch['big_out']
        gt_middle_out = val_batch['middle_out']
        gt_small_out = val_batch['small_out']
        y_true = [gt_big_out, gt_middle_out, gt_small_out]
        gt_big_out_shape = gt_big_out.shape
        gt_middle_out_shape = gt_middle_out.shape
        gt_small_out_shape = gt_small_out.shape

        pred_big_out = self.forward(x)["big_out"]
        pred_middle_out = self.forward(x)["middle_out"]
        pred_small_out = self.forward(x)["small_out"]
        pred_big_out = pred_big_out.reshape(gt_big_out_shape[0], gt_big_out_shape[1], gt_big_out_shape[2],
                                            gt_big_out_shape[3], gt_big_out_shape[4])
        pred_middle_out = pred_middle_out.reshape(gt_middle_out_shape[0], gt_middle_out_shape[1],
                                                  gt_middle_out_shape[2], gt_middle_out_shape[3],
                                                  gt_middle_out_shape[4])
        pred_small_out = pred_small_out.reshape(gt_small_out_shape[0], gt_small_out_shape[1], gt_small_out_shape[2],
                                                gt_small_out_shape[3], gt_small_out_shape[4])
        y_pred = [pred_big_out, pred_middle_out, pred_small_out]

        loss_fn = YoloLoss(self.input_shape, self.classes, self.branch_num, self.anchor, self.batch_size)
        total_loss, _, _, _ = loss_fn(y_pred, y_true)
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_epoch_end(self, outputs):
        pass
