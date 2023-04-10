import torch
from train import DeepNeuralNet
import numpy as np


def main():
  pytorch_model = DeepNeuralNet(n_x = 164, n_h = [64, 32, 8], n_y = 1)
  pytorch_model.load_state_dict(torch.load('pytorch_model.pt'))
  pytorch_model.eval()
  dummy_input = torch.tensor(np.random.randint(2, size=164), dtype=torch.float64)
  dummy_input = torch.reshape(dummy_input, (1, 164))
  print(dummy_input)
  torch.onnx.export(pytorch_model, dummy_input, 'app/public/onnx_model.onnx', verbose=True)


if __name__ == '__main__':
  main()