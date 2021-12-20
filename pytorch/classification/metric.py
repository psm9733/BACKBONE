import numpy as np
import torch

class TopkAccuracy:
    def __init__(self, topk=1):
        self.topk = topk
        self.acc = []

    def calAcc(self, y_true, y_pred):
        batch_size = y_pred.size(0)
        correct_count = 0
        for index in range(batch_size):
            gt_cls = y_true[index].item()
            pred = y_pred[index,:]
            pred_cls_list = torch.argsort(pred, dim=0, descending=True)
            for pred_cls in pred_cls_list[:self.topk]:
                if pred_cls.item() == gt_cls:
                    correct_count += 1
                    break
        acc = correct_count / batch_size
        self.acc.append(acc)

    def getAcc(self):
        return np.array(self.acc).mean()

    def clear(self):
        self.acc = []
