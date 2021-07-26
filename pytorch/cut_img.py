import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob

def cut_img(image_path, target_dir, stride):
    img = cv2.imread(image_path)
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    #sliding window
    h, w, c = img.shape
    if os.path.isdir(target_dir) == False:
        os.mkdir(target_dir)
    if os.path.isdir(target_dir + "/" + img_name) == False:
        os.mkdir(target_dir + "/" + img_name)
    index = 0
    for height in range(0, h, stride):
        for width in range(0, w, stride):
            empty_img = np.zeros((stride, stride, 3), np.uint8)
            crop = img[height:height+stride, width:width+stride, :]
            empty_img[:crop.shape[0], :crop.shape[1], :] = crop
            if np.var(empty_img) > 100:
                cv2.imwrite(target_dir + "/" + img_name + "/" + str(index) + "_" + img_name + ".png", empty_img)
                index += 1


if __name__ == "__main__":
    stride = 256
    input_dir = "/home/fssv1/sangmin/backbone/dataset/lg_noise_remove/train_input_img"
    input_target_dir = "/home/fssv1/sangmin/backbone/dataset/lg_noise_remove/train_input_img_crop"
    label_target_dir = "/home/fssv1/sangmin/backbone/dataset/lg_noise_remove/train_label_img_crop"
    img_list = glob(input_dir + "/*.png")
    for image_path in tqdm(img_list):
        cut_img(image_path, input_target_dir, stride)
        image_path = image_path.replace("input", "label")
        cut_img(image_path, label_target_dir, stride)
