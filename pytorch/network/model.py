from network.densenext.densenext import *
from network.shnet.shnet import *
from network.unet.unet import *
from utils.utils import weight_initialize
from torchsummary import summary
from adamp import *
from utils.metric import TopkAccuracy
from utils.losses import PSNRLoss
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from utils.generator import *
import pytorch_lightning as pl

class Classification(pl.LightningModule):
    def __init__(self, task, batch_size = 1, train_aug = None, val_aug = None, workers = 4, weight_decay = 1e-4):
        super(Classification, self).__init__()
        # hyper parameter
        self.task = task
        self.classes = -1
        self.lr = 1e-3
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
        elif task == "tiny-imagenet":
            self.classes = 200

        # model setting
        self.activation = nn.ReLU()
        self.backbone = DenseNext32(self.activation)
        self.classification_head = nn.Sequential(
            Conv2D_BN(self.backbone.getOutputChannel(), self.activation, 1280, (1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, self.classes, 1)
        )
        self.model = nn.Sequential(
            self.backbone,
            self.classification_head
        )
        weight_initialize(self.model)

    def setup(self, stage):
        if self.task == "mnist":
            self.train_gen = Mnist('S:/sangmin/backbone/dataset/mnist', True, False, self.train_aug, self.classes)
            self.val_gen = Mnist('S:/sangmin/backbone/dataset/mnist', False, False, self.val_aug, self.classes)
        elif self.task == "tiny_imagenet":
            self.train_gen = TinyImageNet('C:/Users/sangmin/Desktop/backbone/dataset/tiny-imagenet-200', True, True, self.train_aug, self.classes)
            self.val_gent = TinyImageNet('C:/Users/sangmin/Desktop/backbone/dataset/tiny-imagenet-200', True, True, self.val_aug, self.classes)

    def forward(self, input):
        output = self.model(input)
        b, c, _, _ = output.size()
        output = output.view(b, c)
        return {"pred": output}

    def configure_optimizers(self):
        self.optimizer = AdamP(self.parameters(), self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=self.lr * 1.2,
                                                      step_size_up=max(0, int(self.trainer.max_epochs / 5)), cycle_momentum=False,
                                                      mode='triangular2', gamma=0.995, verbose=True)
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.worker)

    def val_dataloader(self):
        return DataLoader(self.train_gen, batch_size=max(1, int(self.batch_size / 4)), shuffle=False, drop_last=True, num_workers=self.worker)

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
    def __init__(self, feature_num, input_shape, batch_size = 1, train_aug = None, val_aug = None, workers = 4, weight_decay = 1e-4):
        super(DeNoising, self).__init__()
        activation = nn.PReLU()
        self.lr = 1e-7
        self.input_shape = input_shape
        self.train_aug = train_aug
        self.val_aug = val_aug
        self.batch_size = batch_size
        self.workers = workers
        self.weight_decay = weight_decay
        self.backbone = SHNet(activation, feature_num, groups=32, mode = "bilinear")
        # self.backbone = UNet(3, 3)
        weight_initialize(self.backbone)

    def setup(self, stage):
        self.train_gen = Denoising('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', True, self.input_shape, self.train_aug)
        self.val_gen = Denoising('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', True, self.input_shape, self.val_aug)

    def train_dataloader(self):
        return DataLoader(self.train_gen, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.train_gen, batch_size=max(1, int(self.batch_size / 4)), shuffle=False, drop_last=True, num_workers=self.workers)

    def forward(self, input):
        output = self.backbone(input)
        return {'hg1_loss': output}

    def configure_optimizers(self):
        self.optimizer = AdamP(self.parameters(), self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=self.lr * 1e3,
                                                      step_size_up=max(0, int(self.trainer.max_epochs / 5)), cycle_momentum=False,
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

if __name__ == "__main__":
    activation = nn.PReLU()
    input_shape = (3, 224, 224)
    classes = 200
    model = Classification(activation, classes)
    weight_initialize(model)
    summary(model, input_shape, batch_size=4, device='cpu')