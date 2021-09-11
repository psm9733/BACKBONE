import glob
import tqdm
import numpy as np
import cv2
import os
from random import *

red_list = [(0, 0, 153), (0, 0, 204), (0, 0, 255), (51, 51, 255), (102, 102, 255), (0, 0, 179), (45, 39, 104), (123, 135, 205)]
orange_list = [(0, 191, 230), (128, 212, 255), (0, 170, 255), (0, 136, 204), (0, 128, 255), (0, 153, 230)]
yellow_list = [(51, 255, 255), (25, 255, 255), (0, 255, 255), (102, 255, 255), (153, 255, 255), (204, 255, 255)]
green_list = [(102, 255, 102), (51, 255, 33), (0, 230, 76), (0, 204, 68), (0, 128, 0), (17, 102, 0)]
blue_list = [(255, 0, 0), (255, 42, 21), (255, 0, 0), (247, 67, 12), (240, 60, 15), (235, 68, 20)]
indigo_list = [(120, 20, 0), (110, 18, 0), (110, 20, 2), (122, 22, 2), (140, 26, 18), (97, 18, 12), (24, 12, 0)]
violet_list = [(255, 102, 178), (230, 0, 153), (179, 0, 119), (204, 0, 102), (255, 0, 128), (230, 34, 132)]
white_list = [(255, 255, 255), (252., 252, 252), (250, 250, 250), (248, 248, 248), (245, 245, 245), (240, 240, 240)]
black_list = [(0, 0, 0), (5, 5, 5), (10, 10, 10), (15, 15, 15), (20, 20, 20), (25, 25, 25), (26, 21, 18) ,(18, 5, 3)]
gray_list = [(200, 200, 200), (180, 180, 180), (160, 160, 160), (150, 150, 150)]
beige_list = [(220, 245, 245), (238, 245, 245), (230, 245, 245), (209, 232, 237), (170, 194, 224), (195, 222, 224)]
khaki_list = [(61, 82, 74), (84, 112, 101), (81, 115, 100), (61, 86, 75), (39, 54, 50), (56, 78, 72)]
brown_list = [(71, 112, 191), (50, 83, 143), (95, 131, 199), (41, 70, 124), (36, 63, 111), (28, 48, 85)]
color_list = [red_list, orange_list, yellow_list, green_list, blue_list, indigo_list, violet_list, white_list, black_list, gray_list, beige_list, khaki_list, brown_list]
color_name_list = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'white', 'black', 'gray', 'beige', 'khaki', 'brown']

if __name__ == "__main__":
    train_num_per_color = 2000
    test_num_per_color = 200
    color_num = len(color_list)
    path = glob.glob("../dataset/clothing-detection_dataset/**/*.jpg", recursive = True)
    number = 0
    flag = "train"
    for folder in color_name_list:
        dirpath = "../dataset/color_classification/train/" + folder
        if os.path.isdir(dirpath) == False:
            os.mkdir(dirpath)
    for folder in color_name_list:
        dirpath = "../dataset/color_classification/test/" + folder
        if os.path.isdir(dirpath) == False:
            os.mkdir(dirpath)

    for img_path in tqdm.tqdm(path):
        img_path = img_path.replace("\\", "/")
        txt_path = img_path.replace(".jpg", ".txt")
        img = cv2.imread(img_path)
        with open(txt_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                for index, color in enumerate(color_list):
                    cls, cx, cy, w, h = line.split(" ")
                    x = int((float(cx) - float(w) / 2) * img.shape[1])
                    y = int((float(cy) - float(h) / 2) * img.shape[0])
                    w = int(float(w) * img.shape[1])
                    h = int(float(h) * img.shape[0])
                    crop = img[y:y+h, x:x+w]
                    gen_crop = crop.copy()
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    mask = np.array(cv2.threshold(gray_crop, 200, 255, cv2.THRESH_BINARY_INV))[1]
                    mask = np.where(mask == 255, True, False)

                    color_mask = np.zeros_like(crop)
                    color_select = color[randint(0, len(color) - 1)]
                    gen_crop[:, :, 0] = np.where(mask == True, color_select[0], crop[:, :, 0])
                    gen_crop[:, :, 1] = np.where(mask == True, color_select[1], crop[:, :, 1])
                    gen_crop[:, :, 2] = np.where(mask == True, color_select[2], crop[:, :, 2])
                    # cv2.imshow('color_mask', gen_crop)
                    # cv2.imshow('origin', crop)
                    # cv2.waitKey()
                    if number < train_num_per_color and flag == 'train':
                        cv2.imwrite("../dataset/color_classification/train/" + color_name_list[index%color_num] + "/" + str(number) + ".jpg", gen_crop)
                    elif number < test_num_per_color and flag == 'test':
                        cv2.imwrite("../dataset/color_classification/test/" + color_name_list[index%color_num] + "/" + str(number) + ".jpg", gen_crop)
                    elif number == train_num_per_color and flag == 'train':
                        flag = 'test'
                        number = 0
                    else:
                        exit()
                number += 1

