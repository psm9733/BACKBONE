from network.hourglassnet.hourglassnet import *
from network.regnet.regnet import *
from network.common.blocks import *
from network.common.head import *
from utils.utils import weight_initialize
from adamp import *
from gaze_estimation.losses import WingLoss
from torchvision.utils import make_grid
from gaze_estimation.generator import *
import pytorch_lightning as pl
import torch.nn
from gaze_estimation.config import *
from torch.utils.data import DataLoader


class E3GazeNet(pl.LightningModule):
    def __init__(self, input_shape, batch_size=1, train_aug=None, val_aug=None, workers=4, learning_rate=1e-4,
                 weight_decay=1e-4):
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

        self.stem_in_channels_list = [self.in_channels, 32, 64, 32, 64]
        self.stem_out_channels_list = [32, 64, 32, 64, 128]
        self.stem = StemBlock_5(self.stem_in_channels_list, self.stem_out_channels_list, self.activation, bias=True)
        self.segmentation_backbone = HourglassNet(self.activation, feature_num=self.feature_num, mode="bilinear",
                                                  bias=True)
        self.segmentation_output = Conv2D_BN(self.feature_num, activation=self.activation, out_channels=2,
                                             kernel_size=(3, 3), stride=1, padding='same', bias=True)
        self.regression_backbone = RegNetY_200MF_custom(self.activation, 3,
                                                        bias=True)  # gray channels : 1, seg channels : 2
        self.landmark_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(self.input_shape[1] / self.regression_backbone.getOutputStride() * self.input_shape[
                2] / self.regression_backbone.getOutputStride() * self.regression_backbone.getOutputChannels()),
                      self.output_dense_num)

        )
        weight_initialize(self.stem)
        weight_initialize(self.segmentation_backbone)
        weight_initialize(self.segmentation_output)
        weight_initialize(self.regression_backbone)
        weight_initialize(self.landmark_head)

    def setup(self, stage):
        self.train_gen = EyegazeGenerator(dataset_dir, True, self.train_aug)
        self.val_gen = EyegazeGenerator(dataset_dir, False, self.val_aug)

    def train_dataloader(self):
        return DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True, drop_last=True,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.train_gen, batch_size=self.val_batch_size, shuffle=False, drop_last=True,
                          num_workers=self.workers)

    def forward(self, input):
        output = self.stem(input)
        output = self.segmentation_backbone(output)
        seg_output = self.segmentation_output(output)
        reg_input = torch.cat([seg_output, input], dim=1)
        output = self.regression_backbone(reg_input)[3]
        reg_output = self.landmark_head(output)
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
