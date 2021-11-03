from network.resnext.resnext import *
from network.resnet.resnet import *
from network.densenext.densenext import *
from network.densenet.densenet import *
from network.hourglassnet.hourglassnet import *
from network.unet.unet import *
from network.resnet.resnet import *
from network.regnet.regnet import *
from network.common.blocks import *
from network.common.neck import *
from network.common.head import *
from utils.utils import weight_initialize, get_anchors
from torchsummary import summary
from adamp import *
from utils.metric import TopkAccuracy
from utils.losses import PSNRLoss, WingLoss, AdaptiveWingLoss
from detection.config import *
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from utils.generator import *
from utils.losses import YoloLoss
import pytorch_lightning as pl
import numpy as np
import torch.nn

class Classification(pl.LightningModule):
    def __init__(self, task, batch_size=1, train_aug=None, val_aug=None, workers=4, learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        # hyper parameter
        self.task = task
        self.classes = -1
        self.in_channels = 3
        self.lr = learning_rate
        self.train_aug = train_aug
        self.val_aug = val_aug
        self.batch_size = batch_size
        self.workers = workers
        self.weight_decay = weight_decay
        self.train_acc_top1 = TopkAccuracy(topk=1)
        self.train_acc_top5 = TopkAccuracy(topk=5)
        self.valid_acc_top1 = TopkAccuracy(topk=1)
        self.valid_acc_top5 = TopkAccuracy(topk=5)
        if task == "mnist":
            self.classes = 10
            self.in_channels = 1
        elif task == "tiny_imagenet":
            self.classes = 200

        # model setting
        self.activation = nn.ReLU()
<<<<<<< HEAD
        self.stem_in_channels_list = [self.in_channels, 32, 64, 32, 64]
        self.stem_out_channels_list = [32, 64, 128, 64, 128]
        self.stem = StemBlock_5(self.stem_in_channels_list, self.stem_out_channels_list, self.activation, bias=True)
=======
        self.stem = StemBlock(self.in_channels, self.activation, bias=True)
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        self.backbone = RegNetX_200MF_custom(self.activation, self.stem.getOutputChannels())
        self.classification_head = nn.Sequential(
            Conv2D_BN(self.backbone.getOutputChannels(), self.activation, 1280, (1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, self.classes, 1)
        )
        weight_initialize(self.stem)
        weight_initialize(self.backbone)
        weight_initialize(self.classification_head)

    def setup(self, stage):
        if self.task == "mnist":
            self.train_gen = Mnist('S:/sangmin/backbone/dataset/mnist', True, False, self.train_aug, self.classes)
            self.val_gen = Mnist('S:/sangmin/backbone/dataset/mnist', False, False, self.val_aug, self.classes)
        elif self.task == "tiny_imagenet":
            self.train_gen = TinyImageNet('/home/fssv1/sangmin/backbone/dataset/tiny-imagenet-200', True, False, self.train_aug, self.classes)
            self.val_gen = TinyImageNet('/home/fssv1/sangmin/backbone/dataset/tiny-imagenet-200', False, False, self.val_aug, self.classes)

    def forward(self, input):
        stem_out = self.stem(input)
        backbone_out = self.backbone(stem_out)[3]
        head_out = self.classification_head(backbone_out)
        b, c, _, _ = head_out.size()
        output = head_out.view(b, c)
        return {"pred": output}

    def configure_optimizers(self):
        self.optimizer = AdamP(self.parameters(), self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=self.lr*10,
                                                      step_size_up=10, cycle_momentum=False,
                                                      mode='triangular2', gamma=0.995, verbose=True)
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_gen, batch_size=max(1, int(self.batch_size / 4)), shuffle=False, drop_last=True, num_workers=self.workers)

    def training_step(self, train_batch, batch_idx):
        x = train_batch['img']
        y_true = train_batch['y_true']
        y_pred = self.forward(x)["pred"]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y_true)
        self.train_acc_top1.calAcc(y_true, y_pred)
        self.train_acc_top5.calAcc(y_true, y_pred)
        self.log("lr", self.scheduler.get_lr()[0])
        self.log("top1", self.train_acc_top1.getAcc(), on_step=True, prog_bar=True)
        self.log("top5", self.train_acc_top5.getAcc(), on_step=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch['img']
        y_true = val_batch['y_true']
        y_pred = self.forward(x)["pred"]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y_true)
        self.valid_acc_top1.calAcc(y_true, y_pred)
        self.valid_acc_top5.calAcc(y_true, y_pred)
        self.log("val_top1", self.valid_acc_top1.getAcc(), on_epoch=True, prog_bar=True)
        self.log("val_top5", self.valid_acc_top5.getAcc(), on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.train_acc_top1.clear()
        self.valid_acc_top1.clear()
        self.train_acc_top5.clear()
        self.valid_acc_top5.clear()

class DeNoising(pl.LightningModule):
    def __init__(self, feature_num, input_shape, batch_size=1, train_aug=None, val_aug=None, workers=4, learning_rate=1e-7, weight_decay=1e-4):
        super().__init__()
        self.activation = nn.PReLU()
        self.lr = learning_rate
        self.input_shape = input_shape
        self.train_aug = train_aug
        self.val_aug = val_aug
        self.batch_size = batch_size
        self.workers = workers
        self.weight_decay = weight_decay
        self.stem = Conv2D_BN(self.input_shape[0], activation=self.activation, out_channels=feature_num,
                              kernel_size=(1, 1), stride=1, padding='same')
        self.module1 = HourglassNet(self.activation, feature_num, groups=1, mode = "bilinear")
        self.module1_sam = SAM(self.activation, feature_num, self.input_shape[0], feature_num, kernel_size=(3, 3), stride=1, padding='same', groups = 1)
        # self.backbone = UNet(self.input_shape[0], self.input_shape[0])
        self.model = nn.Sequential(
            self.stem,
            self.module1,
            self.module1_sam
        )
        weight_initialize(self.model)

    def setup(self, stage):
        self.train_gen = DenoisingGenerator('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', self.input_shape, True, self.train_aug)
        self.val_gen = DenoisingGenerator('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', self.input_shape, True, self.val_aug)

    def train_dataloader(self):
        return DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.train_gen, batch_size=max(1, int(self.batch_size / 4)), shuffle=False, drop_last=True, num_workers=self.workers)

    def forward(self, input):
        stem_out = self.stem(input)
        module1_output = self.module1(stem_out)
        module1_output, module1_loss = self.module1_sam(module1_output, input)
        return {'hg1_loss': module1_loss}

    def configure_optimizers(self):
        self.optimizer = AdamP(self.parameters(), self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=self.lr*1e3,
                                                      step_size_up=50, cycle_momentum=False,
                                                      mode='triangular2', gamma=0.995, verbose=True)
        return [self.optimizer], [self.scheduler]

    def training_step(self, train_batch, batch_idx):
        x = train_batch['img']
        y_true = train_batch['y_true']
        y_hg1_pred = self.forward(x)["hg1_loss"]
        y_hg1_pred = torch.clip(y_hg1_pred, 0, 1)
        # y_hg2_pred = self.forward(x)["hg2_loss"]
        loss_fn = PSNRLoss(max_val=1)
        hg1_loss = loss_fn(y_hg1_pred, y_true)
        # hg2_loss = loss_fn(y_hg2_pred, y_true)
        loss=hg1_loss
        self.log("PSNR_loss", loss, on_epoch=True, prog_bar=True)
        grid = make_grid([x[0, :, :, :], y_hg1_pred[0,:, :, :], y_true[0, :, :, :]])
        if (self.global_step % x.size()[0]) == 0:
            self.logger.experiment.add_image("images", grid, self.global_step)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch['img']
        y_true = val_batch['y_true']
        y_hg1_pred = self.forward(x)["hg1_loss"]
        # y_hg2_pred = self.forward(x)["hg2_loss"]
        loss_fn = PSNRLoss(max_val=1)
        hg1_loss = loss_fn(y_hg1_pred, y_true)
        # hg2_loss = loss_fn(y_hg2_pred, y_true)
        loss=hg1_loss
        self.log("val_PSNR_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.log("lr", self.scheduler.get_lr()[0])

class E3GazeNet(pl.LightningModule):
    def __init__(self, input_shape, batch_size = 1, train_aug = None, val_aug = None, workers = 4, learning_rate = 1e-4, weight_decay = 1e-4):
        super().__init__()
        self.activation = nn.ReLU()
        self.lr = learning_rate
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.iris_points = 32
        self.eyelid_points = 16
        self.gaze_vector_num = 3
        self.feature_num = 128
        self.output_dense_num = (self.iris_points + self.eyelid_points) * 2 + self.gaze_vector_num
        self.train_aug = train_aug
        self.val_aug = val_aug
        self.batch_size = batch_size
        self.val_batch_size = max(1, int(self.batch_size / 4))
        self.workers = workers
        self.weight_decay = weight_decay
<<<<<<< HEAD
        self.stem_in_channels_list = [self.in_channels, 32, 64, 32, 64]
        self.stem_out_channels_list = [32, 64, 32, 64, 128]
        self.stem = StemBlock_5(self.stem_in_channels_list, self.stem_out_channels_list, self.activation, bias=True)
        self.segmentation_backbone = HourglassNet(self.activation, feature_num=self.feature_num, mode = "bilinear", bias=True)
        self.segmentation_output = Conv2D_BN(self.feature_num, activation=self.activation, out_channels=2, kernel_size=(3, 3), stride=1, padding='same', bias=True)
        self.regression_backbone = RegNetY_200MF_custom(self.activation, 3, bias=True)     # gray channels : 1, seg channels : 2
        self.landmark_head = nn.Sequential(
            nn.Flatten(),
            # Conv2D_BN(self.regression_backbone.getOutputChannels(), self.activation, 1280, (1, 1)),
            # nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(1280, self.output_dense_num, 1)
            nn.Linear(int(self.input_shape[1] / self.regression_backbone.getOutputStride() * self.input_shape[2] / self.regression_backbone.getOutputStride() * self.regression_backbone.getOutputChannels()), self.output_dense_num)
=======
        self.stem = Conv2D_BN(self.input_shape[0], activation=self.activation, out_channels=self.feature_num, kernel_size=(3, 3), stride=1, padding='same')
        self.segmentation_backbone = HourglassNet(self.activation, feature_num=self.feature_num, mode = "bilinear")
        self.segmentation_output = Conv2D_BN(self.feature_num, activation=self.activation, out_channels=2, kernel_size=(3, 3), stride=1, padding='same')
        self.regression_backbone = RegNetY_200MF_custom(self.activation, 3)     # gray channels : 1, seg channels : 2
        self.landmark_head1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(self.input_shape[1] / self.regression_backbone.getOutputStride() * self.input_shape[2] / self.regression_backbone.getOutputStride() * self.regression_backbone.getOutputChannels()), self.output_dense_num)
        )

        self.landmark_head2 = nn.Sequential(
            Conv2D_BN(self.regression_backbone.getOutputChannels(), self.activation, 1280, (1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, self.output_dense_num, 1)
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        )
        weight_initialize(self.stem)
        weight_initialize(self.segmentation_backbone)
        weight_initialize(self.segmentation_output)
        weight_initialize(self.regression_backbone)
        weight_initialize(self.landmark_head1)
        weight_initialize(self.landmark_head2)

    def setup(self, stage):
        self.train_gen = EyegazeGenerator('/home/fssv1/sangmin/backbone/dataset/unityeyes/train', self.train_aug)
        self.val_gen = EyegazeGenerator('/home/fssv1/sangmin/backbone/dataset/unityeyes/valid', self.val_aug)

    def train_dataloader(self):
        return DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.train_gen, batch_size=self.val_batch_size, shuffle=False, drop_last=True, num_workers=self.workers)

    def forward(self, input):
        output = self.stem(input)
        output = self.segmentation_backbone(output)
        seg_output = self.segmentation_output(output)
        reg_input = torch.cat([seg_output, input], dim=1)
        output = self.regression_backbone(reg_input)[3]
<<<<<<< HEAD
        reg_output = self.landmark_head(output)
=======
        reg_output = self.landmark_head1(output)
>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        return {'seg_pred': seg_output, 'reg_pred': reg_output}

    def configure_optimizers(self):
        self.optimizer = AdamP(self.parameters(), self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=self.lr * 1e1,
                                                      step_size_up=25, cycle_momentum=False,
                                                      mode='triangular2', gamma=0.995, verbose=True)
        return [self.optimizer], [self.scheduler]

    def training_step(self, train_batch, batch_idx):
        img = train_batch['img']
        output = self.forward(img)

        seg_img = train_batch['seg_img']
        iris_landmark = train_batch['iris_landmark']
        eyelid_landmark = train_batch['eyelid_landmark']
        gaze_vector = train_batch['vector']

        seg_pred = output['seg_pred']
        seg_loss_fn = torch.nn.MSELoss()
        seg_loss = seg_loss_fn(seg_pred, seg_img)

        reg_pred = output['reg_pred']
        total_landmark = torch.cat([iris_landmark, eyelid_landmark], dim=1)
        total_landmark = total_landmark.view(self.batch_size, -1)

        lm_reg_pred = reg_pred[:, :(self.iris_points + self.eyelid_points) * 2]
        lm_reg_loss_fn = WingLoss()
        lm_reg_loss = lm_reg_loss_fn(lm_reg_pred, total_landmark)

        vector_reg_pred = reg_pred[:, (self.iris_points + self.eyelid_points) * 2:]
        vector_reg_loss_fn = nn.MSELoss()
        vector_reg_loss = vector_reg_loss_fn(vector_reg_pred, gaze_vector)

        w1 = 5
        w2 = 1
        w3 = int((self.iris_points + self.eyelid_points) / self.gaze_vector_num)
        loss = w1 * seg_loss + w2 * lm_reg_loss + w3 * vector_reg_loss
        if (self.global_step % img.size()[0]) == 0:
            grid = make_grid([seg_img[0, :, :, :], seg_pred[0, :, :, :]])
            self.logger.experiment.add_image("images", grid, self.global_step)

        self.log("seg", w1 * seg_loss, on_epoch=True, prog_bar=True)
        self.log("lm", w2 * lm_reg_loss, on_epoch=True, prog_bar=True)
        self.log("vector", w3 * vector_reg_loss, on_epoch=True, prog_bar=True)
        self.log("loss", loss, on_epoch=True, prog_bar=True)
        return loss

    # def validation_step(self, val_batch, batch_idx):
    #     img = val_batch['img']
    #     output = self.forward(img)
    #
    #     seg_img = val_batch['seg_img']
    #     iris_landmark = val_batch['iris_landmark']
    #     eyelid_landmark = val_batch['eyelid_landmark']
    #     gaze_vector = val_batch['vector']
    #
    #     seg_pred = output['seg_pred']
    #     seg_loss_fn = torch.nn.MSELoss()
    #     seg_loss = seg_loss_fn(seg_pred, seg_img)
    #
    #     reg_pred = output['reg_pred']
    #     total_landmark = torch.cat([iris_landmark, eyelid_landmark], dim=1)
    #     total_landmark = total_landmark.view(self.val_batch_size, -1)
    #
    #     lm_reg_pred = reg_pred[:, :(self.iris_points + self.eyelid_points) * 2]
    #     lm_reg_loss_fn = WingLoss()
    #     lm_reg_loss = lm_reg_loss_fn(lm_reg_pred, total_landmark)
    #
    #     vector_reg_pred = reg_pred[:, (self.iris_points + self.eyelid_points) * 2:]
    #     vector_reg_loss_fn = nn.MSELoss()
    #     vector_reg_loss = vector_reg_loss_fn(vector_reg_pred, gaze_vector)
    #
    #     w1 = 5
    #     w2 = 1
    #     w3 = int((self.iris_points + self.eyelid_points) / self.gaze_vector_num)
    #     loss = w1 * seg_loss + w2 * lm_reg_loss + w3 * vector_reg_loss
    #     self.log("val_loss", loss, on_epoch=True, prog_bar=True)
    #     return loss

    def validation_epoch_end(self, outputs):
        self.log("lr", self.scheduler.get_lr()[0])

class Scaled_Yolov4(pl.LightningModule):
    def __init__(self, input_shape, classes, batch_size=1, train_aug=None, val_aug=None, workers=4, learning_rate=1e-4, weight_decay=1e-4):
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
<<<<<<< HEAD
        self.anchor = get_anchors(ANCHOR_INFO_PATH)
        self.branch_num = 3
        self.anchor_num = int(len(self.anchor) / self.branch_num)
        self.stem_in_channels_list = [self.in_channels, 32, 64]
        self.stem_out_channels_list = [32, 64, 128]
        self.fpn_out_channels_list = [128, 256, 512]
        self.head_out_channels_list = [int(self.anchor_num * (self.classes + 5)), int(self.anchor_num * (self.classes + 5)), int(self.anchor_num * (self.classes + 5))]
        self.stem = StemBlock_3(self.stem_in_channels_list, self.stem_out_channels_list, self.activation, bias=True)
        self.backbone = RegNetX_3stage(self.activation, self.stem.getOutputChannels(), bias=True)
        self.fpn_neck = FPN_3branch(self.backbone.getOutputBranchChannels()[len(self.backbone.getOutputBranchChannels()) - self.branch_num:self.branch_num + 1], self.fpn_out_channels_list, self.activation, bias=True)
        self.yolo_head = Yolo_3branch(self.fpn_neck.getOutputBranchChannels(), self.head_out_channels_list, bias=True)
        self.output_shape = [
            [self.head_out_channels_list[0], int(input_shape[1] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 4), int(input_shape[2] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 4)],
            [self.head_out_channels_list[1], int(input_shape[1] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 2), int(input_shape[2] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 2)],
            [self.head_out_channels_list[2], int(input_shape[1] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 1), int(input_shape[2] / (self.stem.getOutputStride() * self.backbone.getOutputStride()) * 1)],
        ]
=======
        self.stem = StemBlock(self.in_channels, self.activation, bias=True)
        self.backbone = ResNet48(self.activation, self.stem.getOutputChannels())
        self.yolo_head = nn.Sequential(

        )
        self.output_size = [
            [input_shape[1] * self.stem.getOutputStride() * self.backbone.getOutputStride() * 4, input_shape[2] * self.stem.getOutputStride() * self.backbone.getOutputStride() * 4],
            [input_shape[1] * self.stem.getOutputStride() * self.backbone.getOutputStride() * 2, input_shape[2] * self.stem.getOutputStride() * self.backbone.getOutputStride() * 2],
            [input_shape[1] * self.stem.getOutputStride() * self.backbone.getOutputStride() * 1, input_shape[2] * self.stem.getOutputStride() * self.backbone.getOutputStride() * 1],
        ]

>>>>>>> 2ca3a7fdb9861229c7f9c0f4ea14b7641c9b08b1
        weight_initialize(self.stem)
        weight_initialize(self.backbone)
        weight_initialize(self.fpn_neck)
        weight_initialize(self.yolo_head)

    def set_train_path(self, dir_path):
        self.train_dir = dir_path

    def set_valid_path(self, dir_path):
        self.valid_dir = dir_path

    def setup(self, stage):
        self.train_gen = YoloGenerator(self.train_dir, self.input_shape, self.output_shape, self.classes, self.anchor, True, self.train_aug)
        self.val_gen = YoloGenerator(self.valid_dir, self.input_shape, self.output_shape, self.classes, self.anchor, False, self.val_aug)

    def forward(self, input):
        stem_out = self.stem(input)
        big_out, middle_out, small_out = self.backbone(stem_out)[len(self.backbone.getOutputBranchChannels()) - self.branch_num:self.branch_num + 1]
        big_out, middle_out, small_out = self.fpn_neck(big_out, middle_out, small_out)
        big_out, middle_out, small_out = self.yolo_head(big_out, middle_out, small_out)
        return {"big_out": big_out, "middle_out": middle_out, "small_out": small_out}

    def configure_optimizers(self):
        self.optimizer = AdamP(self.parameters(), self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=self.lr*10, step_size_up=20, cycle_momentum=False, mode='triangular2', gamma=0.995, verbose=True)
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_gen, batch_size=max(1, int(self.batch_size / 4)), shuffle=False, drop_last=True, num_workers=self.workers)

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
        pred_big_out = pred_big_out.reshape(gt_big_out_shape[0], gt_big_out_shape[1], gt_big_out_shape[2], gt_big_out_shape[3], gt_big_out_shape[4])
        pred_middle_out = pred_middle_out.reshape(gt_middle_out_shape[0], gt_middle_out_shape[1], gt_middle_out_shape[2], gt_middle_out_shape[3], gt_middle_out_shape[4])
        pred_small_out = pred_small_out.reshape(gt_small_out_shape[0], gt_small_out_shape[1], gt_small_out_shape[2], gt_small_out_shape[3], gt_small_out_shape[4])
        y_pred = [pred_big_out, pred_middle_out, pred_small_out]
        loss_fn = YoloLoss(self.classes, self.branch_num, self.anchor, self.batch_size)
        loss = loss_fn(y_pred, y_true)
        self.log("lr", self.scheduler.get_lr()[0])
        self.log("loss", loss, on_epoch=True, prog_bar=True)
        return loss

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
        pred_big_out = pred_big_out.reshape(gt_big_out_shape[0], gt_big_out_shape[1], gt_big_out_shape[2], gt_big_out_shape[3], gt_big_out_shape[4])
        pred_middle_out = pred_middle_out.reshape(gt_middle_out_shape[0], gt_middle_out_shape[1], gt_middle_out_shape[2], gt_middle_out_shape[3], gt_middle_out_shape[4])
        pred_small_out = pred_small_out.reshape(gt_small_out_shape[0], gt_small_out_shape[1], gt_small_out_shape[2], gt_small_out_shape[3], gt_small_out_shape[4])
        y_pred = [pred_big_out, pred_middle_out, pred_small_out]

        loss_fn = YoloLoss(self.classes, self.branch_num, self.anchor, self.batch_size)
        loss = loss_fn(y_pred, y_true)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        pass

if __name__ == "__main__":
    activation = nn.PReLU()
    input_shape = (1, 224, 224)
    classes = 200
    model = Classification(activation, classes)
    weight_initialize(model)
    summary(model, input_shape, batch_size=4, device='cpu')