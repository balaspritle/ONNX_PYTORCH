#!pip install -U protobuf
#!pip install onnx onnxruntime-gpu

import onnx, cv2, sys
import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession, SessionOptions, get_all_providers
from collections import Counter

class ONNX_Predictor:
    def __init__(self, 
                 #model_path_1=os.path.join(os.path.dirname(__file__), 'ml_models/torch_model/mobilenet-TS.pt'),
                 #model_path_2=os.path.join(os.path.dirname(__file__), 'ml_models/torch_model/resnet-TS.pt'),
                 #model_path_3=os.path.join(os.path.dirname(__file__), 'ml_models/torch_model/efficientnet-b0-TS.pt'), 
                 model_path_1 = '/content/drive/MyDrive/HP-Aviation-Model-Training/ONNX/mobilenet_ocr_bs.onnx',
                 model_path_2 = '/content/drive/MyDrive/HP-Aviation-Model-Training/ONNX/efficientnet-b0-TS1.onnx',
                 model_path_3 = '/content/drive/MyDrive/HP-Aviation-Model-Training/ONNX/resnet-TS.onnx', 
                 img_size = 224, 
                 output_classes = 35):
        self.img_size = img_size
        try:
          self.model_path_1, self.device_1 = self.cpu_or_cuda(model_path_1)
        except:
          self.model_path_1 = None
        
        try:
          self.model_path_2, self.device_2 = self.cpu_or_cuda(model_path_2)
        except:
          self.model_path_2 = None
        
        try:
          self.model_path_3, self.device_3 = self.cpu_or_cuda(model_path_3)
        except:
          self.model_path_3 = None

        self.numbers2text = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                             10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J',
                             19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S',
                             28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}
        
        # First Model
        try:
            print("Model 1 - {} Successfully loaded once on {}".format(
            self.model_path_1.split('/')[-1].split('-')[0].capitalize(), self.device_1))
        except AttributeError:
            print("Model 1 Successfully loaded once on {}".format(self.device_1))
        # Second Model
        if self.model_path_2:
            pass
            try:
                print("Model 2 - {} Successfully loaded once on {}".format(
                self.model_path_2.split('/')[-1].split('-')[0].capitalize(), self.device_2))
            except AttributeError:
                print("Model 2 Successfully loaded once on {}".format(self.device_2))
        # Third Model
        if self.model_path_3:
            pass
            try:
                print("Model 3 - {} Successfully loaded once on {}".format(
                self.model_path_3.split('/')[-1].split('-')[0].capitalize(), self.device_3))
            except AttributeError:
                print("Model 3 Successfully loaded once on {}".format(self.device_3))
    
    def frequent(self, List):
        occurences = Counter(List)
        return occurences.most_common(1)[0][0]
    
    def cpu_or_cuda(self, model_path):
      try:
        gpu_model = self.create_model_for_provider(model_path, "CUDAExecutionProvider")
        return gpu_model, "CUDA:0"
      except:
        cpu_model = self.create_model_for_provider(model_path, "CPUExecutionProvider")
        return cpu_model, "CPU"

    def create_model_for_provider(self, model_path: str, provider: str) -> InferenceSession:
      assert provider in ort.get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
      # Few properties than might have an impact on performances (provided by MS)
      options = SessionOptions()
      options.intra_op_num_threads = 1
      # Load the model as a graph and prepare the CPU backend 
      return InferenceSession(model_path, options, providers=[provider])
    
    def predict_image(self, image_array, majority=True, debug=False):
        # image = cv2.imread(image_array)
        try:
            image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        except:
            image = image_array
        image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32)
        image = np.array(image)
        if debug:
            cv2.imshow('image', image)
            cv2.waitKey(0)
        image /= 255.
        image = np.expand_dims(image, axis=0)
        image = np.moveaxis(image, -1, 1)

        inputs_to_onnx_1 = {self.model_path_1.get_inputs()[0].name: image}
        output_class_1 = self.model_path_1.run(None, inputs_to_onnx_1) 

        if self.model_path_2 and not self.model_path_3:
            inputs_to_onnx_2 = {self.model_path_2.get_inputs()[0].name: image}
            output_class_2 = self.model_path_2.run(None, inputs_to_onnx_2) 
            consolidated_output_class = ((output_class_1[0] + output_class_2[0])/2)

        if self.model_path_2 and self.model_path_3:
            inputs_to_onnx_2 = {self.model_path_2.get_inputs()[0].name: image}
            output_class_2 = self.model_path_2.run(None, inputs_to_onnx_2)
            
            inputs_to_onnx_3 = {self.model_path_3.get_inputs()[0].name: image}
            output_class_3 = self.model_path_3.run(None, inputs_to_onnx_3)
            consolidated_output_class = ((output_class_1[0] + output_class_2[0] + output_class_3[0])/3)
        
        if majority:
            class_list = []
            class_list.append(output_class_1[0].argmax(1).item())
            if self.model_path_2:
                class_list.append(output_class_2[0].argmax(1).item())
            if self.model_path_2 and self.model_path_3:
                class_list.append(output_class_3[0].argmax(1).item())
            major_class = self.frequent(class_list)
            return self.numbers2text[major_class]
        return self.numbers2text[consolidated_output_class.argmax(1).item()]

onnx_predictor = ONNX_Predictor()
response = onnx_predictor.predict_image(cv2.imread('C.png'))
print("response",response)