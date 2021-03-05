!pip install -U protobuf
!pip install onnx onnxruntime-gpu

import onnx, cv2, sys
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession, SessionOptions, get_all_providers ###

model_path = '<Enter full path here>/mobilenet_ocr_bs.onnx'


def loads_multiple_numpy_image():
    return (np.random.randn(10, 3, 224, 224).astype(np.float32)) # (10, 3, 224, 224)

def loads_single_numpy_image(image_path):
    image = cv2.imread(image_path)
    image = image.astype(np.float32)
    image = np.array(image)
    image /= 255.
    image = np.expand_dims(image, axis=0)
    image = np.moveaxis(image, -1, 1)
    return image # (1, 3, 224, 224)
    
numbers2text = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J',
                19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S',
                28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}

def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
  assert provider in ort.get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
  # Few properties than might have an impact on performances (provided by MS)
  options = SessionOptions()
  options.intra_op_num_threads = 1
  # Load the model as a graph and prepare the CPU backend 
  return InferenceSession(model_path, options, providers=[provider])


def cpu_or_cuda():
  try:
    gpu_model = create_model_for_provider(model_path, "CUDAExecutionProvider")
    print("Using CUDA")
    return gpu_model
  except:
    cpu_model = create_model_for_provider(model_path, "CPUExecutionProvider")
    print("Using CPU")
    return cpu_model

image = loads_single_numpy_image() #loads_multiple_numpy_image()

model_device = cpu_or_cuda()

inputs_onnx = {model_device.get_inputs()[0].name: image}
outputs = model_device.run(None, inputs_onnx) ## Here first arguments None becuase we want every output sometimes model return more than one output

for output in outputs[0]:
  print("output", numbers2text[np.array([output]).argmax(1).item()])
