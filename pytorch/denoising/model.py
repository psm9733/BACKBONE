from network.hourglassnet.hourglassnet import *
from network.common.head import *
from utils.utils import weight_initialize
from adamp import *
from denoising.losses import PSNRLoss
from torchvision.utils import make_grid
from denoising.generator import *
from denoising.config import *
import torch
import pytorch_lightning as pl
import torch.nn

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
        self.train_gen = DenoisingGenerator(dataset_dir, self.input_shape, True, self.train_aug)
        self.val_gen = DenoisingGenerator(dataset_dir, self.input_shape, True, self.val_aug)

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