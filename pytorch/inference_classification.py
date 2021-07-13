import torch
import cv2
from glob import glob
from tqdm import tqdm
from network.model import *
from torchvision import transforms
from torchsummary import summary

# Load saved model
PATH = "./saved_model/DenseNext32_no_aug/model_429999.pth"
input_shape = (3, 64, 64)
model = Classification(nn.ReLU(), 200)
model.load_state_dict(torch.load(PATH))
model.eval()
summary(model, input_shape, batch_size=1, device='cpu')

# Prediction
img_paths = glob("./dataset/tiny-imagenet-200/val/**/*.JPEG", recursive=True)
for img_path in tqdm(img_paths):
    img = cv2.imread(img_path)
    cv2.imshow("img", img)
    input_tensor = transforms.ToTensor()(img)
    input_tensor = torch.unsqueeze(input_tensor, 0)
    result = model(input_tensor)
    result = torch.argmax(result['pred'])
    print(img_path)
    print(result)
    cv2.waitKey()