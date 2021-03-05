import onnx, torch, cv2, sys
import numpy as np
# pip install -U protobuf
'''
onnx_model = onnx.load("model_artifacts/mobilenet_ocr.onnx")
onnx.checker.check_model(onnx_model)
# Print a human readable representation of the graph
onnx.helper.printable_graph(onnx_model.graph)
'''
import onnxruntime as ort
#print(ort.get_device(), ort.get_all_providers())
model_path = 'model_artifacts/mobilenet_ocr_bs.onnx'
ort_session = ort.InferenceSession(model_path)

numbers2text = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J',
                19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S',
                28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}

def loads_single_numpy_image():
    image = cv2.imread('C.png')
    image = image.astype(np.float32)
    image = np.array(image)
    image /= 255.
    image = np.expand_dims(image, axis=0)
    image = np.moveaxis(image, -1, 1)
    return image

image = loads_single_numpy_image()

'''
METHOD 1
x = torch.randn(1, 3, 224, 224, requires_grad=True)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_inputs = {ort_session.get_inputs()[0].name: image}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]
print(numbers2text[img_out_y.argmax(1).item()])
'''

# METHOD 2 - CPU
from onnxruntime import InferenceSession, SessionOptions, get_all_providers ###
import onnxruntime as ort

def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
  assert provider in ort.get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
  # Few properties than might have an impact on performances (provided by MS)
  options = SessionOptions()
  options.intra_op_num_threads = 1
  # Load the model as a graph and prepare the CPU backend 
  return InferenceSession(model_path, options, providers=[provider])

cpu_model = create_model_for_provider(model_path, "CPUExecutionProvider")
#gpu_model = create_model_for_provider(model_path, "CUDAExecutionProvider")

#inputs_onnx= {'input':image}
inputs_onnx = {ort_session.get_inputs()[0].name: image}
output = cpu_model.run(None, inputs_onnx) ## Here first arguments None becuase we want every output sometimes model return more than one output
#output = gpu_model.run(None, inputs_onnx) ## Here first arguments None becuase we want every output sometimes model return more than one output

img_out_y = output[0]
print(numbers2text[img_out_y.argmax(1).item()])

# METHOD 3 - (GPU)
# !pip install onnxruntime-gpu #for gpu

