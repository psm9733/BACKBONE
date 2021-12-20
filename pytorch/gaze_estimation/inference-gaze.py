import os
from model import *
from torchvision import transforms
import tqdm
import random
import torch
import warnings
from utils.utils import unityeye_json_process
from datetime import datetime


def DrawPoint(img, points, color):
    new_img = img.copy()
    for index in range(0, len(points)):
        if index % 2 == 0:
            new_img = cv2.circle(new_img, (int(points[index]), int(points[index + 1])), radius=1, color=color)
    return new_img


def Predict(img, model):
    bgr_img = img.copy()
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    input_tensor = transforms.ToTensor()(gray_img.astype(np.float32).copy() / 255.)
    input_tensor = torch.unsqueeze(input_tensor, 0)
    result = model(input_tensor)
    seg_img = result['seg_pred']
    reg_out = result['reg_pred']
    reg_out = torch.squeeze(reg_out)
    reg_out = reg_out.detach().numpy()
    iris_lm = reg_out[0:64]
    eyelid_lm = reg_out[64:96]
    gaze_vector = reg_out[96:99]
    gaze_vector[1] = -gaze_vector[1]

    eye_c = np.mean(iris_lm.reshape(-1, 2), axis=0).astype(int)
    bgr_img = cv2.line(bgr_img.copy(), tuple(eye_c), tuple(eye_c + (gaze_vector[:2] * 80).astype(int)), (0, 0, 255), 2)
    lm_img = DrawPoint(bgr_img, iris_lm, (0, 255, 0))
    lm_img = DrawPoint(lm_img, eyelid_lm, (0, 255, 255))

    seg_img = torch.squeeze(seg_img)
    seg_img = torch.permute(seg_img, (1, 2, 0))
    seg_img = seg_img.detach().numpy() * 255
    cv2.imshow('iris', seg_img[:, :, 0])
    cv2.moveWindow('iris', 228, 200)
    cv2.imshow('eyelid', seg_img[:, :, 1])
    cv2.moveWindow('eyelid', 356, 200)

    lm_img = cv2.resize(lm_img, (256, 256))
    gray_img = cv2.resize(gray_img, (256, 256))
    filename = datetime.today().strftime("%Y%m%d%H%M%S")
    # cv2.imwrite(filename + '_input.jpg', gray_img)
    # cv2.imwrite(filename + '_result.jpg', lm_img)
    cv2.imshow('lm', lm_img)
    cv2.moveWindow('lm', 484, 200)
    cv2.waitKey()


if __name__ == "__main__":
    engine = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if engine == torch.device('cpu'):
        warnings.warn('Cannot use CUDA context. Train might be slower!')
    # Load saved model
    PATH = "R:/sangmin/backbone/saved_model/20210816021412/epoch=32_val_loss=2.9075.ckpt"
    bbox_hoffset = [38, 40, 44, 48, 52, 50, 60]
    bbox_woffset = [57, 60, 66, 72, 78, 60, 72]
    input_shape = (1, 96, 128)
    model = E3GazeNet(input_shape=input_shape)
    model.load_state_dict(torch.load(PATH)['state_dict'])
    model.eval()
    # summary(model, input_shape, batch_size=1, device='cpu')

    # Prediction
    img_paths = glob.glob("R:/sangmin/backbone/dataset/unityeyes/test/**/*.jpg", recursive=True)
    random.shuffle(img_paths)
    for index, img_path in enumerate(tqdm.tqdm(img_paths)):
        img = cv2.imread(img_path)
        json_path = img_path.replace('.jpg', '.json')
        json_file = open(json_path)
        json_data = json.load(json_file)
        gt_iris_landmark = np.array(unityeye_json_process(img, json_data['iris_2d']))
        gt_eyelid_landmark = np.array(unityeye_json_process(img, json_data['interior_margin_2d']))
        gt_gaze_vector = np.array(eval(json_data['eye_details']['look_vec'])).astype(np.float32)
        xmin = np.min(gt_eyelid_landmark[:, 0]) - bbox_woffset[index % len(bbox_hoffset)]
        xmax = np.max(gt_eyelid_landmark[:, 0]) + bbox_woffset[index % len(bbox_hoffset)]
        ymin = np.min(gt_eyelid_landmark[:, 1]) - bbox_hoffset[index % len(bbox_hoffset)]
        ymax = np.max(gt_eyelid_landmark[:, 1]) + bbox_hoffset[index % len(bbox_hoffset)]
        bbox = [xmin, ymin, xmax, ymax]
        crop_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        crop_img = cv2.resize(crop_img, (input_shape[2], input_shape[1]))
        Predict(crop_img, model)
