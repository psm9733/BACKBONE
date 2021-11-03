import glob
import random
import os
from tqdm import tqdm
import cv2

count = 0
is_txt = 0
txt_dst = "../dataset_info/train.txt"
train_list = list(set(glob.glob("/home/fssv1/sangmin/backbone/dataset/coco/train2017/**/*.jpg", recursive=True)))
valid_list = list(set(glob.glob("/home/fssv1/sangmin/backbone/dataset/coco/val2017/**/*.jpg", recursive=True)))
with open("../dataset_info/train.txt", "w") as f:
	print("start make train.txt")
	fcount = 0
	for img_path in tqdm(train_list):
		is_txt = 0
		txt_path = img_path.replace(".jpg",".txt")
		if os.path.isfile(txt_path):
			count += 1
			img_path = img_path.replace("\\", '/')
			ctc = img_path
			img = cv2.imread(img_path, cv2.IMREAD_COLOR)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img_h, img_w, img_c = img.shape
			if os.path.exists(txt_path) == True:
				with open(txt_path, "r") as bbox:
					bbox_line = bbox.readlines()
					if len(bbox_line) == 0:
						ctc += "\n"
					else:
						for index in range(len(bbox_line)):
							b_line = bbox_line[index]
							cls, cx, cy, w, h = b_line.rstrip('\n').split(" ")
							x1 = float(cx) - float(w) / 2
							y1 = float(cy) - float(h) / 2
							x2 = float(cx) + float(w) / 2
							y2 = float(cy) + float(h) / 2
							ctc += " " + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + "," + cls
							if index == len(bbox_line) - 1:
								ctc += "\n"
				if fcount == 0:
					f = open(txt_dst, "w")
					f.write(ctc)
					f.close()
				else:
					f = open(txt_dst, "a")
					f.write(ctc)
					f.close()
				fcount += 1

with open("../dataset_info/valid.txt", "w") as f:
	print("start make valid.txt")
	for img in tqdm(valid_list):
		is_txt = 0
		txt_root = img.replace(".jpg",".txt")
		if os.path.isfile(txt_root):
			f.write(img + "\n")
			count += 1