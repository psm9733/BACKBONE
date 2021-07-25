import torch
import torch.nn as nn
from network.common.layers import Conv2D_BN
from network.resnet.resnet import *
from network.resnext.resnext import *
from network.densenet.densenet import *
from network.densenext.densenext import *
from network.shnet.shnet import *
from network.unet.unet import *
from utils.utils import weight_initialize
from torchsummary import summary
import pytorch_lightning as pl
from adamp import *
from utils.metric import TopkAccuracy
from utils.losses import PSNRLoss
from torchvision.utils import make_grid

class Classification(pl.LightningModule):
    def __init__(self, classes, min_lr=1e-4, max_lr = 1e-3, weight_decay = 1e-4):
        super(Classification, self).__init__()
        self.activation = nn.PReLU()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.backbone = DenseNext32(self.activation)
        self.classification_head = nn.Sequential(
            Conv2D_BN(self.backbone.getOutputChannel(), self.activation, 1280, (1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )
        self.model = nn.Sequential(
            self.backbone,
            self.classification_head
        )
        self.train_acc_top1 = TopkAccuracy(topk=1)
        self.train_acc_top5 = TopkAccuracy(topk=5)
        self.valid_acc_top1 = TopkAccuracy(topk=1)
        self.valid_acc_top5 = TopkAccuracy(topk=5)
        weight_initialize(self.model)

    def forward(self, input):
        output = self.model(input)
        b, c, _, _ = output.size()
        output = output.view(b, c)
        return {"pred": output}

    def configure_optimizers(self):
        self.optimizer = AdamP(self.parameters(), self.min_lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.min_lr, max_lr=self.max_lr,
                                                      step_size_up=max(0, int(self.trainer.max_epochs / 5)), cycle_momentum=False,
                                                      mode='triangular2', gamma=0.995, verbose=True)
        return [self.optimizer], [self.scheduler]


    def training_step(self, train_batch, batch_idx):
        x = train_batch['img']
        y_true = train_batch['y_true']
        y_pred = self.forward(x)["pred"]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y_true)
        self.train_acc_top1.calAcc(y_true, y_pred)
        self.train_acc_top5.calAcc(y_true, y_pred)
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
        return loss

    def validation_epoch_end(self, outputs):
        self.train_acc_top1.clear()
        self.valid_acc_top1.clear()
        self.train_acc_top5.clear()
        self.valid_acc_top5.clear()
        self.log("lr", self.scheduler.get_lr()[0])

class DeNoising(pl.LightningModule):
    def __init__(self, feature_num, min_lr=1e-4, max_lr = 1e-3, weight_decay = 1e-4):
        super(DeNoising, self).__init__()
        activation = nn.PReLU()
        groups = 32
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.backbone = SHNet(activation, feature_num, groups=groups, mode = "")
        weight_initialize(self.backbone)

    def forward(self, input):
        output1, output2 = self.backbone(input)
        return {'hg1_loss': output1, 'hg2_loss': output2}

    def configure_optimizers(self):
        self.optimizer = AdamP(self.parameters(), self.min_lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.min_lr, max_lr=self.max_lr,
                                                      step_size_up=max(0, int(self.trainer.max_epochs / 5)), cycle_momentum=False,
                                                      mode='triangular2', gamma=0.995, verbose=True)
        return [self.optimizer], [self.scheduler]

    def training_step(self, train_batch, batch_idx):
        x = train_batch['img']
        y_true = train_batch['y_true']
        y_hg1_pred = self.forward(x)["hg1_loss"]
        y_hg2_pred = self.forward(x)["hg2_loss"]
        loss_fn = PSNRLoss(max_val=1)
        hg1_loss = loss_fn(y_hg1_pred, y_true)
        hg2_loss = loss_fn(y_hg2_pred, y_true)
        loss=hg1_loss+hg2_loss
        self.log("PSNR_loss", loss, on_epoch=True, prog_bar=True)
        grid = make_grid([x[0, :, :, :], y_hg2_pred[0,:, :, :], y_true[0, :, :, :]])
        if (self.global_step % x.size()[0]) == 0:
            self.logger.experiment.add_image("images", grid, self.global_step)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch['img']
        y_true = val_batch['y_true']
        y_hg1_pred = self.forward(x)["hg1_loss"]
        y_hg2_pred = self.forward(x)["hg2_loss"]
        loss_fn = PSNRLoss(max_val=1)
        hg1_loss = loss_fn(y_hg1_pred, y_true)
        hg2_loss = loss_fn(y_hg2_pred, y_true)
        loss=hg1_loss+hg2_loss
        self.log("val_PSNR_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.log("lr", self.scheduler.get_lr()[0])

if __name__ == "__main__":
    activation = nn.PReLU()
    input_shape = (3, 224, 224)
    classes = 200
    model = Classification(activation, classes)
    weight_initialize(model)
    summary(model, input_shape, batch_size=4, device='cpu')