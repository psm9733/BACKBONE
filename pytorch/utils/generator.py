import numpy as np
from albumentations.core.serialization import load
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations
import albumentations.pytorch
from torch.utils.data import DataLoader
import tqdm
import torch
import torch.nn.functional as F
import glob
import os
import cv2

class TinyImageNet(Dataset):
    def __init__(self, root_dir, is_train, one_hot, transform=None, num_classes = 200) -> None:
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
        self.num_classes = num_classes
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

class NoiseReduction(Dataset):
    def __init__(self, root_dir, is_train, input_shape, transform=None, num_classes=200) -> None:
        super(NoiseReduction, self).__init__()
        self.transform = transform
        self.is_train = is_train
        self.input_shape = input_shape
        if is_train:
            self.data = glob.glob(root_dir + '/train_input_img/**/*.png', recursive=True)
            self.train_list = dict()
            for data in self.data:
                data = data.replace("\\", "/")
                label = data.replace("train_input_img", "train_label_img").replace("input", "label")
                self.train_list[data] = label
        else:
            self.data = glob.glob(root_dir + '/train_input_img/**/*.png', recursive=True)
            self.train_list = dict()
            for data in self.data:
                data = data.replace("\\", "/")
                label = data.replace("train_input_img", "train_label_img").replace("input", "label")
                self.train_list[data] = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_imgpath = self.data[index].replace("\\", "/")
        gt_imgpath = self.train_list[input_imgpath]
        input_img = cv2.imread(input_imgpath)
        input_img = cv2.resize(input_img, (self.input_shape[1], self.input_shape[2]))
        gt_img = cv2.imread(gt_imgpath)
        gt_img = cv2.resize(gt_img, (self.input_shape[1], self.input_shape[2]))
        if self.transform:
            transformed = self.transform(image=input_img, image1=input_img, image2=gt_img)
            x = transformed['image1']
            y = transformed['image2']
        else:
            x = transforms.ToTensor(input_img)
            y = transforms.ToTensor(gt_img)
        return {'img': x, 'y_true': y}


if __name__ == '__main__':
    input_shape = (3, 1280, 720)
    transform = albumentations.Compose([
        albumentations.Resize(height=input_shape[1], width=input_shape[2]),
        albumentations.Normalize(0, 1),
        albumentations.SomeOf([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(-90, 90),
            albumentations.RandomResizedCrop(height=input_shape[1], width=input_shape[2]),
            # albumentations.ColorJitter(),
        ], 3, p=0.5),
        # albumentations.OneOf([
        #     albumentations.MotionBlur(p=1),
        #     albumentations.OpticalDistortion(p=1),
        #     albumentations.GaussNoise(p=1)
        # ], p=0.5),
        albumentations.pytorch.ToTensorV2(),
    ], additional_targets={'image1': 'image', 'image2': 'image'})

    # loader = DataLoader(TinyImageNet('C:/Users/sangmin/Desktop/backbone/dataset/tiny-imagenet-200', True, True, transform), batch_size=128, shuffle=True, num_workers=4)
    loader = DataLoader(NoiseReduction('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', True, input_shape, transform), batch_size=1, num_workers=1)
    for i in range(2):
        print('epoch', i)
        for batch, sample in tqdm.tqdm(enumerate(loader), total=loader.__len__()):
            img = sample['img'].numpy()[0]
            gt_img = sample['y_true'].numpy()[0]
            img = np.transpose(img, (1, 2, 0))
            gt_img = np.transpose(gt_img, (1, 2, 0))
            img = cv2.resize(img, (input_shape[1], input_shape[2]))
            gt_img = cv2.resize(gt_img, (input_shape[1], input_shape[2]))
            cv2.imshow("input_img", img)
            cv2.imshow("gt_img", gt_img)
            cv2.waitKey()