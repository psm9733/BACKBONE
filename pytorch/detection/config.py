ANCHOR_INFO_PATH = "C:/Users/sangmin/Desktop/backbone/pytorch/detection/dataset_info/yolo_anchors.txt"
TRAIN_DIR_PATH = "C:/Users/sangmin/Desktop/backbone/dataset/coco/train2017"
VALID_DIR_PATH = "C:/Users/sangmin/Desktop/backbone/dataset/coco/val2017"

backbone_name = "Regnet-yolo"
input_shape = (3, 608, 608)
batch_size = 32
learning_rate = 1e-3
weight_decay = 5e-4
classes = 80
TRAINING_IOU_THRESHOLD = 0.5
max_epochs = 125
workers = 1
