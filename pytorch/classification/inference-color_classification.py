import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import cv2
import numpy as np
import glob
from network.model import *
from torchvision import transforms
import tqdm
from torchsummary import summary
import random
import pytorch_lightning as pl
import warnings

engine = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if engine == torch.device('cpu'):
    warnings.warn('Cannot use CUDA context. Train might be slower!')

# Load saved model
PATH = "C:/Users/sangmin/Desktop/backbone/pytorch/saved_model/20210831231640/epoch=9_val_loss=0.1289.ckpt"
# PATH = "./resave_DenseNext32.pth"
input_shape = (3, 224, 224)
model = Classification(task = "color_mnist")
model.load_state_dict(torch.load(PATH)["state_dict"])
model.eval()
# summary(model, input_shape, batch_size=1, device='cpu')

# Prediction
color_list = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'white', 'black', 'gray', 'beige', 'khaki', 'brown']
img_paths = glob.glob("C:/Users/sangmin/Desktop/backbone/dataset/clothing-detection_dataset/**/*.jpg", recursive=True)
random.shuffle(img_paths)
for img_path in tqdm.tqdm(img_paths):
    img = cv2.imread(img_path)
    txt_path = img_path.replace(".jpg", ".txt")
    with open(txt_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            cls, cx, cy, w, h = line.split(" ")
            x = int((float(cx) - float(w) / 2) * img.shape[1])
            y = int((float(cy) - float(h) / 2) * img.shape[0])
            w = int(float(w) * img.shape[1])
            h = int(float(h) * img.shape[0])
            crop = img[y:y + h, x:x + w]
            input_tensor = transforms.ToTensor()(crop.astype(np.float32) / 255.)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            result = model(input_tensor)["pred"]
            result = torch.argmax(result)
            print(color_list[result.data])
            cv2.imshow("crop", crop)
            cv2.waitKey()