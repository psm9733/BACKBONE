import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations.pytorch
from torch.utils.data import DataLoader
import tqdm
import glob
import cv2


class DenoisingGenerator(Dataset):
    def __init__(self, dataset_dir, input_shape, is_train, transform=None) -> None:
        super().__init__()
        """
        : dataset_dir: data path
        : is_train: set whether load dataset as trainset or valid set
        : input_shape: Model input shape
        : transform: albumentation transforms
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.is_train = is_train
        self.input_shape = input_shape
        self.load_data()

    def load_data(self):
        if self.is_train:
            self.data = glob.glob(self.dataset_dir + '/train_input_img_crop/**/*.png', recursive=True)
        else:
            self.data = glob.glob(self.dataset_dir + '/train_input_img_crop/**/*.png', recursive=True)
        self.train_list = dict()
        for data in self.data:
            data = data.replace("\\", "/")
            label = data.replace("train_input_img_crop", "train_label_img_crop").replace("input", "label")
            self.train_list[data] = label

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
    input_shape = (3, 608, 608)
    loader_denoise = True
    transform = albumentations.Compose([
        albumentations.Resize(height=input_shape[1], width=input_shape[2]),
        # albumentations.Normalize(0, 1),
        albumentations.SomeOf([
            # albumentations.Sharpen(),
            # albumentations.Affine(),
            # albumentations.ColorJitter(),
            # albumentations.ToGray(),
            # albumentations.RandomResizedCrop(height=input_shape[1], width=input_shape[2]),
            # albumentations.ColorJitter(),
            albumentations.Flip()
        ], 3, p=1),
        albumentations.OneOf([
            albumentations.Sharpen(p=1),
            albumentations.MotionBlur(p=1),
            # albumentations.Emboss(p=1),
            # albumentations.Equalize(p=1),
        ], p=0.5),
        albumentations.pytorch.ToTensorV2(),
    ])
    if loader_denoise:
        loader = DataLoader(
            DenoisingGenerator('/home/fssv1/sangmin/backbone/dataset/lg_noise_remove', input_shape, True, transform),
            batch_size=1, num_workers=1)

    for i in range(2):
        print('epoch', i)
        for batch, sample in tqdm.tqdm(enumerate(loader), total=loader.__len__()):
            if batch > 100:
                break
            if loader_denoise:
                img = sample['img'].numpy()[0]
                gt_img = sample['y_true'].numpy()[0]
                img = np.transpose(img, (1, 2, 0))
                gt_img = np.transpose(gt_img, (1, 2, 0))
                img = cv2.resize(img, (input_shape[1], input_shape[2]))
                gt_img = cv2.resize(gt_img, (input_shape[1], input_shape[2]))
                cv2.imshow("input_img", img)
                cv2.imshow("gt_img", gt_img)
                cv2.waitKey()
