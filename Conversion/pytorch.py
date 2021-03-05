from torch import nn
import torch.onnx
import numpy as np
import pandas as pd
import torch, os, cv2, torchvision, math
from torchvision import models
from collections import Counter
import os, sys, subprocess

numbers2text = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J',
                19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S',
                28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}

model_path_1 = 'model_artifacts/mobilenet-TS.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft_1 = torch.load(model_path_1, map_location=device)
model_ft_1 = model_ft_1['model'].to(device).eval()
image = cv2.imread('C.png')
image = image.astype(np.float32)
image = np.array(image)
image /= 255.
image = torch.from_numpy(image)
image = image.to(device)
image = image.unsqueeze(0).permute(0, 3, 1, 2)
print("Input Shape", image.shape)
output_class_1 = model_ft_1(image)
print("output_class_1 Shape", output_class_1.shape)
output = output_class_1.argmax(1).item()
print(numbers2text[output])

# for single input prediction
#torch.onnx.export(model_ft_1, image, "model_artifacts/efficientnet-b0-TS.onnx")

# For batch input prediction
torch.onnx.export(model_ft_1, image, "model_artifacts/resnet-TS.onnx", export_params=True, opset_version=10,
                  do_constant_folding=True, input_names = ['input'], output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})