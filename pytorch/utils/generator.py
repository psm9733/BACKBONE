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
import json
import natsort
from utils.utils import splitDataset, unityeye_json_process

class Mnist(Dataset):
    def __init__(self, root_dir, is_train, one_hot, transform=None, num_classes = 10) -> None:
        """
        : root_dir: data path
        : is_train: set whether load dataset as trainset or valid set
        : one_hot: if True: y = [0, 0, 0, ..., 1, 0, 0, ..., ] as keras.uitls.to_categorical
                   else: y = [class_idx] for torch.nn.CrossEntropyLoss
        : transform: albumentation transforms
        """
        super(Mnist, self).__init__()
        self.transform = transform
        self.one_hot = one_hot
        self.is_train = is_train
        self.num_classes = num_classes

        if self.is_train:
            self.image_list = glob.glob(root_dir + "/train/**/*.jpg", recursive=True)
        else:
            self.image_list = glob.glob(root_dir + "/test/**/*.jpg", recursive=True)
        self.label_list = dict()

        for img_path in self.image_list:
            label = img_path.split(os.sep)[-2]
            self.label_list[img_path] = label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_file = self.image_list[index]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.
        if self.transform:
            x = self.transform(image=img)['image']
        else:
            x = transforms.ToTensor(img)
        label = int(self.label_list[img_file])
        if self.one_hot:
            y = F.one_hot(torch.tensor(label), self.num_classes)
        else:
            y = torch.tensor(label)
        return {'img': x, 'y_true': y}

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
        img = img.astype(np.float32) / 255.
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

class EyegazeGenerator(Dataset):
    def __init__(self, root_dir, transform=None) -> None:
        super(EyegazeGenerator, self).__init__()
        self.transform = transform
        self.image_list = natsort.natsorted(glob.glob(root_dir + "/**/*.jpg", recursive=True))
        self.bbox_hoffset = [42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
        self.label_list = []
        for img_path in self.image_list:
            label = img_path.replace(".jpg", ".json")
            self.label_list.append(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        img = cv2.imread(image_path)
        json_path = self.label_list[index]
        json_file = open(json_path)
        json_data = json.load(json_file)
        eyelid_landmark = np.array(unityeye_json_process(img, json_data['interior_margin_2d']))
        iris_landmark = np.array(unityeye_json_process(img, json_data['iris_2d']))
        gaze_vector = np.array(eval(json_data['eye_details']['look_vec'])).astype(np.float32)
        ymin = np.min(eyelid_landmark[:, 1]) - self.bbox_hoffset[index % len(self.bbox_hoffset)]
        ymax = np.max(eyelid_landmark[:, 1]) + self.bbox_hoffset[index % len(self.bbox_hoffset)]
        h = ymax - ymin
        w = np.max(eyelid_landmark[:, 0]) - np.min(eyelid_landmark[:, 0])
        resize_w = h * (img.shape[1] / img.shape[0])
        offset_w = (resize_w - w) / 2
        xmin = np.min(eyelid_landmark[:, 0]) - offset_w
        xmax = np.max(eyelid_landmark[:, 0]) + offset_w
        bbox = [xmin, ymin, xmax, ymax]
        new_iris_landmark = np.array([[int(point3d[0] - xmin), int(point3d[1] - ymin)] for point3d in iris_landmark])
        new_eyelid_landmark = np.array([[int(point3d[0] - xmin), int(point3d[1] - ymin)] for point3d in eyelid_landmark])
        crop_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].astype(np.float32) / 255.
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        if self.transform:
            landmark = []
            landmark.extend(new_iris_landmark)
            landmark.extend(new_eyelid_landmark)
            transform = self.transform(image=crop_img, keypoints=landmark)
            crop_img = transform['image']
            landmark = np.array(transform['keypoints']).astype(np.int32)
            new_iris_landmark = landmark[:new_iris_landmark.shape[0],:]
            new_eyelid_landmark = landmark[new_iris_landmark.shape[0]:,:]
        else:
            crop_img = transforms.ToTensor(crop_img)
        seg_img = np.zeros((2, crop_img.shape[1], crop_img.shape[2]), dtype=np.uint8)
        cv2.fillConvexPoly(seg_img[0], new_iris_landmark, 1)
        cv2.fillConvexPoly(seg_img[1], new_eyelid_landmark, 1)

        seg_img = torch.Tensor(seg_img)
        new_iris_landmark = torch.Tensor(new_iris_landmark.astype(np.float32))
        new_eyelid_landmark = torch.Tensor(new_eyelid_landmark.astype(np.float32))
        gaze_vector = torch.Tensor(gaze_vector)
        return {"img":crop_img, "seg_img":seg_img, 'iris_landmark':new_iris_landmark, 'eyelid_landmark': new_eyelid_landmark, 'vector':gaze_vector[:3]}

class DenoisingGenerator(Dataset):
    def __init__(self, root_dir, is_train, input_shape, transform=None, num_classes=200) -> None:
        super(DenoisingGenerator, self).__init__()
        self.transform = transform
        self.is_train = is_train
        self.input_shape = input_shape
        if is_train:
            self.data = glob.glob(root_dir + '/train_input_img_crop/**/*.png', recursive=True)
            self.train_list = dict()
            for data in self.data:
                data = data.replace("\\", "/")
                label = data.replace("train_input_img_crop", "train_label_img_crop").replace("input", "label")
                self.train_list[data] = label
        else:
            self.data = glob.glob(root_dir + '/train_input_img_crop/**/*.png', recursive=True)
            self.train_list = dict()
            for data in self.data:
                data = data.replace("\\", "/")
                label = data.replace("train_input_img_crop", "train_label_img_crop").replace("input", "label")
                self.train_list[data] = label
        # if is_train:
        #     self.data = glob.glob(root_dir + '/train_input_img_test/**/*.png', recursive=True)
        #     self.train_list = dict()
        #     for data in self.data:
        #         data = data.replace("\\", "/")
        #         label = data.replace("train_input_img_test", "train_label_img_test").replace("input", "label")
        #         self.train_list[data] = label
        # else:
        #     self.data = glob.glob(root_dir + '/train_input_img_test/**/*.png', recursive=True)
        #     self.train_list = dict()
        #     for data in self.data:
        #         data = data.replace("\\", "/")
        #         label = data.replace("train_input_img_test", "train_label_img_test").replace("input", "label")
        #         self.train_list[data] = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_imgpath = self.data[index].replace("\\", "/")
        gt_imgpath = self.train_list[input_imgpath]
        input_img = cv2.imread(input_imgpath)
        input_img = cv2.resize(input_img, (self.input_shape[1], self.input_shape[2]))
        # input_img = input_img.astype(np.float32)
        gt_img = cv2.imread(gt_imgpath)
        gt_img = cv2.resize(gt_img, (self.input_shape[1], self.input_shape[2]))
        # gt_img = gt_img.astype(np.float32)
        if self.transform:
            transformed = self.transform(image=input_img, image1=input_img, image2=gt_img)
            x = transformed['image1']
            y = transformed['image2']
        else:
            x = transforms.ToTensor(input_img)
            y = transforms.ToTensor(gt_img)
        return {'img': x, 'y_true': y}


if __name__ == '__main__':
    input_shape = (1, 96, 128)
    transform = albumentations.Compose([
        albumentations.Resize(height=input_shape[1], width=input_shape[2]),
        # albumentations.Normalize(0, 1),
        albumentations.SomeOf([
            # albumentations.Sharpen(),
            # albumentations.Affine(),
            # albumentations.ColorJitter(),
            # albumentations.Rotate(-90, 90),
            # albumentations.ToGray()
            # albumentations.RandomResizedCrop(height=input_shape[1], width=input_shape[2]),
            # albumentations.ColorJitter(),
        ], 3, p=0.5),
        albumentations.OneOf([
            albumentations.Sharpen(p=1),
            albumentations.MotionBlur(p=1),
            # albumentations.Emboss(p=1),
            # albumentations.Equalize(p=1),
        ], p=0.5),
        albumentations.pytorch.ToTensorV2(),
    ], keypoint_params=albumentations.KeypointParams(format='xy'))
    # ], additional_targets={'image1': 'image', 'image2': 'image'})

    # loader = DataLoader(Mnist('S:/sangmin/backbone/dataset/mnist', True, True, transform), batch_size=128, shuffle=True, num_workers=1)
    # loader = DataLoader(TinyImageNet('C:/Users/sangmin/Desktop/backbone/dataset/tiny-imagenet-200', True, True, transform), batch_size=128, shuffle=True, num_workers=4)
    # loader = DataLoader(DenoisingGenerator('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', True, input_shape, transform), batch_size=1, num_workers=1)
    loader = DataLoader(EyegazeGenerator('R:/sangmin/backbone/dataset/unityeyes/train', transform), batch_size=1, shuffle=True, num_workers=1)
    for i in range(2):
        print('epoch', i)
        for batch, sample in tqdm.tqdm(enumerate(loader), total=loader.__len__()):
            # img = sample['img'].numpy()[0]
            # gt_img = sample['y_true'].numpy()[0]
            # img = np.transpose(img, (1, 2, 0))
            # gt_img = np.transpose(gt_img, (1, 2, 0))
            # img = cv2.resize(img, (input_shape[1], input_shape[2]))
            # gt_img = cv2.resize(gt_img, (input_shape[1], input_shape[2]))
            # cv2.imshow("input_img", img)
            # cv2.imshow("gt_img", gt_img)
            # cv2.waitKey()

            img = sample['img'].numpy()[0]
            seg_img = sample['seg_img'].numpy()[0]
            iris_landmark = sample['iris_landmark'].numpy()[0]
            eyelid_landmark = sample['eyelid_landmark'].numpy()[0]
            gaze_vector = sample['vector'].numpy()[0]
            img = np.transpose(img, (1, 2, 0)) * 255
            img = img.astype(np.uint8)
            gaze_vector[1] = -gaze_vector[1]
            eye_c = np.mean(iris_landmark[:, :2], axis=0).astype(int)
            img = cv2.line(img.copy(), tuple(eye_c), tuple(eye_c + (gaze_vector[:2] * 80).astype(int)), (255, 255, 255), 2)
            for ldmk in np.array(iris_landmark):
                cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (255, 255, 255), -1)
            for ldmk in np.array(eyelid_landmark):
                cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (255, 255, 255), -1)
            cv2.imshow('img', img)
            cv2.imshow('seg1_img', seg_img[0])
            cv2.imshow('seg2_img', seg_img[1])
            cv2.waitKey()