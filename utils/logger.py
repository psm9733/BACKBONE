from torch.utils.tensorboard import SummaryWriter
import torch
class Logger:
    def __init__(self, logdir, log_freq, save_freq):
        self.writer = SummaryWriter(logdir, flush_secs=1)
        self.logdir = logdir
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.iters = 0

    def step(self, data_dict, model):
        if self.iters % self.log_freq == self.log_freq - 1:
            for k, v in data_dict.items():
                self.writer.add_scalar(k, v, self.iters)

        if self.iters % self.save_freq == self.save_freq - 1:
            torch.save(model.state_dict(),
                       '{}/model_{}.pth'.format(self.logdir, self.iters))
        self.iters += 1

    def add_scalar(self, tag, x, y):
        self.writer.add_scalar(tag, y, x)
