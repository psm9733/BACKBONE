import torch

PATH = "./20210706110417/model_30999.pth"

model = torch.load(PATH)
model.eval()