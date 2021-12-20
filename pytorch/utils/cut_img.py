import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob


def cut_img(image_path, input_target_dir, label_target_dir, stride):
    input_img = cv2.imread(image_path)
    input_img_name = os.path.splitext(os.path.basename(image_path))[0]
    label_img_path = image_path.replace("input", "label")
    label_img = cv2.imread(label_img_path)
    label_img_name = os.path.splitext(os.path.basename(label_img_path))[0]
    # sliding window
    h, w, c = input_img.shape
    if os.path.isdir(input_target_dir) == False:
        os.mkdir(input_target_dir)
    if os.path.isdir(input_target_dir + "/" + input_img_name) == False:
        os.mkdir(input_target_dir + "/" + input_img_name)
    if os.path.isdir(label_target_dir) == False:
        os.mkdir(label_target_dir)
    if os.path.isdir(label_target_dir + "/" + label_img_name) == False:
        os.mkdir(label_target_dir + "/" + label_img_name)
    index = 0
    for height in range(0, h, stride):
        for width in range(0, w, stride):
            empty_img = np.zeros((stride, stride, 3), np.uint8)
            crop = input_img[height:height + stride, width:width + stride, :]
            empty_img[:crop.shape[0], :crop.shape[1], :] = crop
            if np.var(empty_img) > 100:
                cv2.imwrite(input_target_dir + "/" + input_img_name + "/" + str(index) + "_" + input_img_name + ".png",
                            empty_img)
                empty_img = np.zeros((stride, stride, 3), np.uint8)
                crop = label_img[height:height + stride, width:width + stride, :]
                empty_img[:crop.shape[0], :crop.shape[1], :] = crop
                cv2.imwrite(label_target_dir + "/" + label_img_name + "/" + str(index) + "_" + label_img_name + ".png",
                            empty_img)
                index += 1


if __name__ == "__main__":
    stride = 256
    input_dir = "/home/fssv1/sangmin/backbone/dataset/lg_noise_remove/train_input_img"
    input_target_dir = "/home/fssv1/sangmin/backbone/dataset/lg_noise_remove/train_input_img_crop"
    label_target_dir = "/home/fssv1/sangmin/backbone/dataset/lg_noise_remove/train_label_img_crop"
    img_list = glob(input_dir + "/*.png")
    for image_path in tqdm(img_list):
        cut_img(image_path, input_target_dir, label_target_dir, stride)
