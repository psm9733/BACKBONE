import cv2
from tqdm import tqdm
from glob import glob

if __name__ == "__main__":
    img_list = "./log_img/**/*.jpg"
    img_list = glob(img_list, recursive = True)
    for img_path in tqdm(img_list):
        img_path = img_path.replace("\\", "/")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", img)
        cv2.waitKey()