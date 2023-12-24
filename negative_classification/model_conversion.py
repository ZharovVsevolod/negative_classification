from model import SpecificBERT_Lightning
import torch

model = SpecificBERT_Lightning.load_from_checkpoint("outputs/2023-12-24/16-18-51/weights/epoch=9-step=100.ckpt")

save_onnx = "negative_classification/models/self_model.onnx"
input_sample = torch.randint(low=0, high=300, size=(1, 60))
model.to_onnx(save_onnx, input_sample, export_params=True)