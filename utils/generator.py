from albumentations.core.serialization import load
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn.functional as F
import glob
import os
import cv2

class TinyImageNet(Dataset):
    num_classes = 200

    def __init__(self, root_dir, is_train, one_hot, transform=None) -> None:
        """
        : root_dir: data path
        : is_train: set whether load dataset as trainset or valid set
        : one_hot: if True: y = [0, 0, 0, ..., 1, 0, 0, ..., ] as keras.uitls.to_categorical
                   else: y = [class_idx] for torch.nn.CrossEntropyLoss
        : transform: albumentation transforms

        """
        super(TinyImageNet, self).__init__()
        self.transform = transform
        self.one_hot = one_hot
        self.is_train = is_train
        with open(root_dir + '/wnids.txt', 'r') as f:
            self.label_list = f.read().splitlines()
        if is_train:
            self.data = glob.glob(root_dir + '/train/*/images/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-3]
                self.train_list[data] = self.label_list.index(label)

        else:
            self.data = glob.glob(root_dir + '/val/images/*.JPEG')
            self.val_list = dict()
            with open(root_dir + '/val/val_annotations.txt', 'r') as f:
                val_labels = f.read().splitlines()
                for label in val_labels:
                    f_name, label, _, _, _, _ = label.split('\t')
                    self.val_list[f_name] = self.label_list.index(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        img = cv2.imread(img_file)
        if self.transform:
            x = self.transform(image=img)['image']
        else:
            x = transforms.ToTensor(img)

        if self.is_train:
            label = self.train_list[img_file]
        else:
            label = self.val_list[os.path.basename(img_file)]

        if self.one_hot:
            y = F.one_hot(torch.tensor(label), self.num_classes)
        else:
            y = torch.tensor(label)
        return {'img': x, 'y_true': y}


if __name__ == '__main__':
    import albumentations
    import albumentations.pytorch
    from torch.utils.data import DataLoader
    import numpy as np
    import tqdm

    transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ])

    loader = DataLoader(TinyImageNet(
        'E:/FSNet2/Datasets/tiny-imagenet-200', True, True, transform), batch_size=128, shuffle=True, num_workers=4)
    for i in range(2):
        print('epoch', i)
        for batch, sample in tqdm.tqdm(enumerate(loader), total=loader.__len__()):
            img = sample['img'].numpy()[0]
