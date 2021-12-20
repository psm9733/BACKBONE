import numpy as np
from torch.utils.data import Dataset
import albumentations.pytorch
from torch.utils.data import DataLoader
import tqdm
import torch
import glob
import cv2
import logging
from detection.config import *
from utils.utils import read_anchors


class YoloGenerator(Dataset):
    def __init__(self, dataset_dir, input_shape, output_shape, classes, anchors, is_train, transform=None) -> None:
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
        self.output_shape = output_shape
        self.classes = classes
        self.load_data()
        self.anchors = anchors
        self.epsilon = 1e-7

    def load_data(self):
        self.data = glob.glob(self.dataset_dir + '/**/*.jpg', recursive=True)

    def __len__(self):
        return len(self.data)

    def get_iou(self, img_shape, anchor, target):
        '''
            args:
                img_shape:(c, h, w)
                anchor:[cx, cy, w, h](0 ~ 1, 0 ~ 1, 0 ~ 1, int)
                target_wh:[cx, cy, w, h](0 ~ 1, 0 ~ 1, 0 ~ 1, 0 ~ 1, float)
        '''
        cx1, cy1, w1, h1 = anchor
        cx2, cy2, w2, h2 = target

        # box = (x1, y1, x2, y2)
        box1 = [cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2]
        box2 = [cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2]
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou

    def __getitem__(self, index):
        # variable init
        y_true = []
        branch_num = len(self.output_shape)
        anchor_num_per_branch = int(len(self.anchors) / branch_num)
        for branch_index in range(branch_num):
            filter_width, h, w = self.output_shape[branch_index]
            grid = np.zeros((anchor_num_per_branch, int(filter_width / anchor_num_per_branch), h, w), dtype=np.float32)
            y_true.append(grid)

        img_path = self.data[index]
        img = cv2.imread(img_path)
        txt_path = img_path.replace(".jpg", ".txt")
        bboxes = []
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                cls, cx, cy, w, h = line.split(" ")
                cx = int(float(cx) * img.shape[1])
                cy = int(float(cy) * img.shape[0])
                w = int(float(w) * img.shape[1])
                h = int(float(h) * img.shape[0])
                x_min = max(0, cx - w / 2)
                y_min = max(0, cy - h / 2)
                if w != 0 and h != 0:
                    bboxes.append([x_min, y_min, w, h, cls])
        if self.transform:
            transformed = self.transform(image=img, bboxes=bboxes)
            transformed_img = transformed['image'] / 255.
            transformed_bboxes = transformed['bboxes']
        else:
            logging.error("Please add transform arg")
        transformed_bboxes = np.array(transformed_bboxes, dtype=np.int32)
        # show_img = np.transpose(transformed_img.numpy(), (1, 2, 0))
        for bbox in transformed_bboxes:
            max_iou = 0
            select_iou_anchor_index = -1
            select_grid_y = -1
            select_grid_x = -1
            select_bbox = []
            '''
                select_bbox:
                    [conf, cx, cy, w, h, cls]
            '''
            select_branch_index = -1
            x_min, y_min, w, h, cls = bbox
            cx = (x_min + w / 2) / transformed_img.shape[2]
            cy = (y_min + h / 2) / transformed_img.shape[1]
            w = w / transformed_img.shape[2]
            h = h / transformed_img.shape[1]
            for branch_index in range(branch_num):
                grid = y_true[branch_index]
                '''
                    grid:
                        [anchor, cls, cy, cx]
                '''
                _, _, grid_h, grid_w = grid.shape
                grid_x = int(cx * (grid_w - 1))
                grid_y = int(cy * (grid_h - 1))
                cell_w = int(transformed_img.shape[2] / grid_w)
                cell_h = int(transformed_img.shape[1] / grid_h)
                for anchor_index in range(anchor_num_per_branch):
                    anchor = self.anchors[anchor_num_per_branch * (branch_num - branch_index - 1) + anchor_index]
                    iou = self.get_iou(transformed_img.shape, [grid_x / grid_w, grid_y / grid_h, anchor[0], anchor[1]],
                                       [cx, cy, w, h])
                    if (iou > TRAINING_IOU_THRESHOLD) and (max_iou < iou):
                        max_iou = iou
                        select_iou_anchor_index = anchor_index
                        select_grid_y = grid_y
                        select_grid_x = grid_x
                        select_bbox = [1.0, ((x_min + w / 2) % cell_w) / cell_w, ((y_min + h / 2) % cell_h) / cell_h, w,
                                       h, 1.0]
                        select_branch_index = branch_index
            if max_iou > TRAINING_IOU_THRESHOLD:
                select_grid = y_true[select_branch_index]
                select_grid[select_iou_anchor_index][0][select_grid_y][select_grid_x] = select_bbox[0]
                select_grid[select_iou_anchor_index][1][select_grid_y][select_grid_x] = select_bbox[1]
                select_grid[select_iou_anchor_index][2][select_grid_y][select_grid_x] = select_bbox[2]
                select_grid[select_iou_anchor_index][3][select_grid_y][select_grid_x] = select_bbox[3]
                select_grid[select_iou_anchor_index][4][select_grid_y][select_grid_x] = select_bbox[4]
                select_grid[select_iou_anchor_index][5 + int(cls)][select_grid_y][select_grid_x] = select_bbox[5]
        # for branch_index in range(branch_num):
        #     grid = y_true[branch_index]
        #     for anchor_index in range(anchor_num_per_branch):
        #         conf = grid[anchor_index, 0, :, :] * 255
        #         cv2.imshow(str(self.output_shape[branch_index]) + "_" + str(anchor_index) + "_conf", cv2.resize(conf, (608, 608), interpolation=cv2.INTER_NEAREST))
        # cv2.imshow("show_img", show_img)
        # cv2.waitKey()
        y_true_big = torch.Tensor(y_true[0])
        y_true_middle = torch.Tensor(y_true[1])
        y_true_small = torch.Tensor(y_true[2])
        return {'img': transformed_img, 'big_out': y_true_big, 'middle_out': y_true_middle, 'small_out': y_true_small}


if __name__ == '__main__':
    input_shape = (3, 608, 608)
    loader_Yolo = True
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
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    if loader_Yolo:
        loader = DataLoader(YoloGenerator('S:/sangmin/backbone/dataset/coco/train2017', input_shape,
                                          [[255, 64, 64], [255, 32, 32], [255, 16, 16]], 80,
                                          read_anchors(ANCHOR_INFO_PATH), True, transform))
    for i in range(2):
        print('epoch', i)
        for batch, sample in tqdm.tqdm(enumerate(loader), total=loader.__len__()):
            if batch > 100:
                break

            if loader_Yolo:
                pass
