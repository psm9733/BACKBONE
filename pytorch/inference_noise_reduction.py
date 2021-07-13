import torch
import cv2
from glob import glob
from tqdm import tqdm
from network.model import *
from torchvision import transforms
from torchsummary import summary
import numpy as np
# Load saved model
PATH = "./saved_model/20210709113556/model_3999.pth"
input_shape = (3, 3264, 2440)
feature_num = 24
model = Segmentation(nn.ReLU(), feature_num)
model.load_state_dict(torch.load(PATH))
model.eval()
summary(model, input_shape, batch_size=1, device='cpu')

# Prediction
img_paths = glob("/home/fssv1/sangmin/backbone/dataset/lg_noise_remove/test_input_img/**/*.png", recursive=True)
for img_path in tqdm(img_paths):
    img = cv2.imread(img_path)
    show_origin = cv2.resize(img, (1024, 720))
    input_tensor = transforms.ToTensor()(img.astype(np.float32) / 255)
    input_tensor = torch.unsqueeze(input_tensor, 0)
    result = model(input_tensor)
    result = result['pred']
    result = torch.squeeze(result)
    result = torch.permute(result, (1, 2, 0))
    pre_show = cv2.resize(result.detach().numpy(), (1024, 720))
    # pre_show *= 255.
    cv2.imshow("img", show_origin)
    cv2.imshow("pre", pre_show)
    cv2.waitKey()