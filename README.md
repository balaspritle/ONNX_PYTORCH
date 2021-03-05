# ONNX_PYTORCH
Contains the code to convert pytorch model to onnx and also deployment.
Also it switches to GPU backend if available, otherwise uses CPU backend.

## CONVERSION

Contains the codes for converting the pytorch based models to onnx format. Models are located at "model_artifacts" folder.

## Deployment

1. Contains the generic code ("methods.py") to deploy an onnx converted model.

2. Contains the actual OCR Implementation function (1. ONNX_Predictor.py) --> ONNX_Predictor instead of TorchPredictor to remove the pytorch dependencies.

