from torch.utils.tensorboard import SummaryWriter
import torch

class Saver:
    def __init__(self, savedir, save_freq):
        self.save_freq = save_freq
        self.iters = 0
        self.savedir = savedir

    def step(self, model):
        if self.iters % self.save_freq == self.save_freq - 1:
            torch.save(model.state_dict(),'{}/model_{}.pth'.format(self.savedir, self.iters))
        self.iters += 1
