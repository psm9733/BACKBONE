from network.regnet.regnet import *
from network.common.blocks import *
from network.common.head import *
from utils.utils import weight_initialize
from adamp import *
from torchsummary import summary
from classification.metric import TopkAccuracy
from classification.generator import *
import pytorch_lightning as pl
import torch.nn
from classification.config import *
from torch.utils.data import DataLoader

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
        self.stem_in_channels_list = [self.in_channels, 32, 64, 128, 64]
        self.stem_out_channels_list = [32, 64, 128, 64, 128]
        self.stem = StemBlock_5(self.stem_in_channels_list, self.stem_out_channels_list, self.activation, bias=True)
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
            self.train_gen = Mnist(mnist_dataset_dir, True, False, self.train_aug, self.classes)
            self.val_gen = Mnist(mnist_dataset_dir, False, False, self.val_aug, self.classes)
        elif self.task == "tiny_imagenet":
            self.train_gen = TinyImageNet(tiny_imagenet_dir, True, False, self.train_aug, self.classes)
            self.val_gen = TinyImageNet(tiny_imagenet_dir, False, False, self.val_aug, self.classes)

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

if __name__ == "__main__":
    input_shape = (3, 224, 224)
    classes = 200
    model = Classification("tiny_imagenet", classes)
    weight_initialize(model)
    summary(model, input_shape, batch_size=4, device='cpu')