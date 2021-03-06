import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations.pytorch
from torch.utils.data import DataLoader
import tqdm
import torch
import glob
import cv2
import json
from utils.utils import unityeye_json_process


class EyegazeGenerator(Dataset):
    def __init__(self, dataset_dir, is_train, transform=None) -> None:
        super().__init__()
        """
        : dataset_dir: data path
        : is_train: set whether load dataset as trainset or valid set
        : one_hot: if True: y = [0, 0, 0, ..., 1, 0, 0, ..., ] as keras.uitls.to_categorical
                   else: y = [class_idx] for torch.nn.CrossEntropyLoss
        : transform: albumentation transforms
        """
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        self.transform = transform
        self.bbox_hoffset = [42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
        self.label_list = []
        self.load_data()

    def load_data(self):
        if self.is_train:
            self.image_list = glob.glob(self.dataset_dir + "/train/**/*.jpg", recursive=True)
        else:
            self.image_list = glob.glob(self.dataset_dir + "/valid/**/*.jpg", recursive=True)

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
        new_eyelid_landmark = np.array(
            [[int(point3d[0] - xmin), int(point3d[1] - ymin)] for point3d in eyelid_landmark])
        crop_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].astype(np.float32) / 255.
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        if self.transform:
            landmark = []
            landmark.extend(new_iris_landmark)
            landmark.extend(new_eyelid_landmark)
            transform = self.transform(image=crop_img, keypoints=landmark)
            crop_img = transform['image']
            landmark = np.array(transform['keypoints']).astype(np.int32)
            new_iris_landmark = landmark[:new_iris_landmark.shape[0], :]
            new_eyelid_landmark = landmark[new_iris_landmark.shape[0]:, :]
        else:
            crop_img = transforms.ToTensor(crop_img)
        seg_img = np.zeros((2, crop_img.shape[1], crop_img.shape[2]), dtype=np.uint8)
        cv2.fillConvexPoly(seg_img[0], new_iris_landmark, 1)
        cv2.fillConvexPoly(seg_img[1], new_eyelid_landmark, 1)

        seg_img = torch.Tensor(seg_img)
        new_iris_landmark = torch.Tensor(new_iris_landmark.astype(np.float32))
        new_eyelid_landmark = torch.Tensor(new_eyelid_landmark.astype(np.float32))
        gaze_vector = torch.Tensor(gaze_vector)
        return {"img": crop_img, "seg_img": seg_img, 'iris_landmark': new_iris_landmark,
                'eyelid_landmark': new_eyelid_landmark, 'vector': gaze_vector[:3]}


if __name__ == '__main__':
    input_shape = (3, 608, 608)
    loader_Eyegaze = True
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
    ], keypoint_params=albumentations.KeypointParams(format='xy'))

    if loader_Eyegaze:
        loader = DataLoader(EyegazeGenerator('S:/sangmin/backbone/dataset/unityeyes/train', transform), batch_size=1,
                            shuffle=True, num_workers=1)

    for i in range(2):
        print('epoch', i)
        for batch, sample in tqdm.tqdm(enumerate(loader), total=loader.__len__()):
            if batch > 100:
                break

            if loader_Eyegaze:
                img = sample['img'].numpy()[0]
                seg_img = sample['seg_img'].numpy()[0]
                iris_landmark = sample['iris_landmark'].numpy()[0]
                eyelid_landmark = sample['eyelid_landmark'].numpy()[0]
                gaze_vector = sample['vector'].numpy()[0]
                img = np.transpose(img, (1, 2, 0)) * 255
                img = img.astype(np.uint8)
                gaze_vector[1] = -gaze_vector[1]
                eye_c = np.mean(iris_landmark[:, :2], axis=0).astype(int)
                img = cv2.line(img.copy(), tuple(eye_c), tuple(eye_c + (gaze_vector[:2] * 80).astype(int)),
                               (255, 255, 255), 2)
                for ldmk in np.array(iris_landmark):
                    cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (255, 255, 255), -1)
                for ldmk in np.array(eyelid_landmark):
                    cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (255, 255, 255), -1)
                cv2.imshow('img', img)
                cv2.imshow('seg1_img', seg_img[0])
                cv2.imshow('seg2_img', seg_img[1])
                cv2.waitKey()
