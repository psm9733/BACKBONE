import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import cv2
import glob
import tqdm
from network.model import *
from torchvision import transforms
from torchsummary import summary
import numpy as np
import torch.nn as nn
from network.model import DeNoising
import natsort

def predict(model, img_path, input_shape, stride=128):
    img = cv2.imread(img_path)
    result_img = np.zeros_like(img).astype(np.float)
    voting_mask = np.zeros_like(img)
    for h in tqdm.tqdm(range(0, img.shape[0], stride)):
        for w in tqdm.tqdm(range(0, img.shape[1], stride)):
            empty_img = np.zeros((input_shape[1], input_shape[2], input_shape[0]), np.uint8)
            crop = img[h:h+input_shape[1], w:w+input_shape[2],:]
            empty_img[:crop.shape[0], :crop.shape[1],:] = crop
            input_tensor = transforms.ToTensor()(empty_img.astype(np.float32) / 255)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            result = model(input_tensor)
            result = result['hg1_loss']
            result = torch.squeeze(result)
            result = torch.permute(result, (1, 2, 0))
            result = result.detach().numpy() * 255
            result_img[h:h+crop.shape[0], w:w+crop.shape[1],:] += result[:crop.shape[0], :crop.shape[1],:]
            voting_mask[h:h+crop.shape[0], w:w+crop.shape[1],:] += 1
    result_img = result_img / voting_mask
    result_img = result_img.astype(np.uint8)
    # cv2.imshow("origin", img)
    # cv2.imshow("result", result_img)
    # cv2.waitKey()
    return result_img

if __name__ == "__main__":
    # Load saved model
    PATH = "saved_model/20210729174036/UNet.ckpt"
    input_shape = (3, 256, 256)
    output_shape = (3, 3264, 2448)
    feature_num = 384
    model = DeNoising(feature_num = feature_num, input_shape=input_shape)
    model.load_state_dict(torch.load(PATH)["state_dict"])
    model.eval()
    summary(model, input_shape, batch_size=1, device='cpu')
    # img_paths = glob.glob("/home/fssv1/sangmin/backbone/dataset/lg_noise_remove/test_input_img/**/*.png", recursive=True)
    img_paths = glob.glob("S:/sangmin/backbone/dataset/lg_noise_remove/test_input_img/**/*.png", recursive=True)
    for img_path in tqdm.tqdm(natsort.natsorted(img_paths, reverse=True)):
        result = predict(model, img_path, input_shape)
        save_path = img_path.replace("test_input_img", "sample_submission").replace("_input", "")
        cv2.imwrite(save_path, result)