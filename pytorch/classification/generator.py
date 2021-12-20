import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn.functional as F
import glob
import os
import cv2
import random

class Mnist(Dataset):
    def __init__(self, dataset_dir, is_train, one_hot, transform=None, num_classes = 10) -> None:
        """
        : root_dir: data path
        : is_train: set whether load dataset as trainset or valid set
        : one_hot: if True: y = [0, 0, 0, ..., 1, 0, 0, ..., ] as keras.uitls.to_categorical
                   else: y = [class_idx] for torch.nn.CrossEntropyLoss
        : transform: albumentation transforms
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.one_hot = one_hot
        self.is_train = is_train
        self.num_classes = num_classes
        self.load_data()

    def load_data(self):
        if self.is_train:
            self.image_list = glob.glob(self.dataset_dir + "/train/**/*.jpg", recursive=True)
            random.shuffle(self.image_list)
        else:
            self.image_list = glob.glob(self.dataset_dir + "/test/**/*.jpg", recursive=True)
        self.label_list = dict()

        for img_path in self.image_list:
            label = img_path.split(os.sep)[-2]
            self.label_list[img_path] = label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.
        if self.transform:
            x = self.transform(image=img)['image']
        else:
            x = transforms.ToTensor(img)
        label = int(self.label_list[img_path])
        if self.one_hot:
            y = F.one_hot(torch.tensor(label), self.num_classes)
        else:
            y = torch.tensor(label)
        return {'img': x, 'y_true': y}

class ColorMnist(Dataset):
    def __init__(self, dataset_dir, is_train, one_hot, color_list, transform=None, num_classes = 10) -> None:
        """
        : dataset_dir: data path
        : is_train: set whether load dataset as trainset or valid set
        : one_hot: if True: y = [0, 0, 0, ..., 1, 0, 0, ..., ] as keras.uitls.to_categorical
                   else: y = [class_idx] for torch.nn.CrossEntropyLoss
        : transform: albumentation transforms
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.color_list = color_list
        self.transform = transform
        self.one_hot = one_hot
        self.is_train = is_train
        self.num_classes = num_classes
        self.load_data()

    def load_data(self):
        if self.is_train:
            self.image_list = glob.glob(self.dataset_dir + "/train/**/*.jpg", recursive=True)
        else:
            self.image_list = glob.glob(self.dataset_dir + "/test/**/*.jpg", recursive=True)
        self.label_list = dict()

        for img_path in self.image_list:
            label = img_path.split(os.sep)[-2]
            self.label_list[img_path] = self.color_list.index(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.
        if self.transform:
            x = self.transform(image=img)['image']
        else:
            x = transforms.ToTensor(img)
        label = int(self.label_list[img_path])
        if self.one_hot:
            y = F.one_hot(torch.tensor(label), self.num_classes)
        else:
            y = torch.tensor(label)
        return {'img': x, 'y_true': y}

class TinyImageNet(Dataset):
    def __init__(self, dataset_dir, is_train, one_hot, transform=None, num_classes = 200) -> None:
        """
        : dataset_dir: data path
        : is_train: set whether load dataset as trainset or valid set
        : one_hot: if True: y = [0, 0, 0, ..., 1, 0, 0, ..., ] as keras.uitls.to_categorical
                   else: y = [class_idx] for torch.nn.CrossEntropyLoss
        : transform: albumentation transforms
        """
        super().__init__()
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.one_hot = one_hot
        self.is_train = is_train
        self.num_classes = num_classes
        self.load_data()

    def load_data(self):
        with open(self.dataset_dir + '/wnids.txt', 'r') as f:
            self.label_list = f.read().splitlines()
        if self.is_train:
            self.data = glob.glob(self.dataset_dir + '/train/*/images/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-3]
                self.train_list[data] = self.label_list.index(label)

        else:
            self.data = glob.glob(self.dataset_dir + '/val/images/*.JPEG')
            self.val_list = dict()
            with open(self.dataset_dir + '/val/val_annotations.txt', 'r') as f:
                val_labels = f.read().splitlines()
                for label in val_labels:
                    f_name, label, _, _, _, _ = label.split('\t')
                    self.val_list[f_name] = self.label_list.index(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = cv2.imread(img_path)
        img = img.astype(np.float32) / 255.
        if self.transform:
            x = self.transform(image=img)['image']
        else:
            x = transforms.ToTensor(img)

        if self.is_train:
            label = self.train_list[img_path]
        else:
            label = self.val_list[os.path.basename(img_path)]

        if self.one_hot:
            y = F.one_hot(torch.tensor(label), self.num_classes)
        else:
            y = torch.tensor(label)
        return {'img': x, 'y_true': y}