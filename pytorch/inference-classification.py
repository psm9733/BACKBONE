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
PATH = "./saved_model/20210731151442/epoch=4_val_loss=0.2201.ckpt"
# PATH = "./resave_DenseNext32.pth"
input_shape = (1, 28, 28)
model = Classification(task = "mnist")
model.load_state_dict(torch.load(PATH)["state_dict"])
model.eval()
# summary(model, input_shape, batch_size=1, device='cpu')

# Prediction
img_paths = glob.glob("./dataset/mnist/test/**/*.jpg", recursive=True)
random.shuffle(img_paths)
for img_path in tqdm.tqdm(img_paths):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img", img)
    input_tensor = transforms.ToTensor()(img.astype(np.float32) / 255.)
    input_tensor = torch.unsqueeze(input_tensor, 0)
    result = model(input_tensor)["pred"]
    result = torch.argmax(result)
    print(result)
    cv2.waitKey()