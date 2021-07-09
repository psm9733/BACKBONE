import torch

class TopKAccuracy:
    def __init__(self, one_hot, topK=[1, 5]) -> None:
        self.one_hot = one_hot
        self.topK = topK
        self.samples = 0
        self.accuracy = dict()
        for k in self.topK:
            self.accuracy[k] = 0

    def step(self, y_pred, y_true):
        maxK = max(self.topK)
        if self.one_hot:
            pass
        else:
            pred = torch.topk(y_pred, maxK, 1).indices
            pred = pred.t()
            correct = pred.eq(y_true.view(1, -1).expand_as(pred))
            for k in self.topK:
                acc = correct[:k].unsqueeze(0).sum()
                self.accuracy[k] = (
                    self.accuracy[k] * self.samples + acc) / (self.samples + y_pred.size()[0])
        self.samples += y_pred.size()[0]

    def clear(self):
        self.samples = 0
        self.accuracy = dict()
        for k in self.topK:
            self.accuracy[k] = 0
