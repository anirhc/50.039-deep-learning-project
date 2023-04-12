import torch
import model as m
import numpy as np


def main():
  """
  This function loads a PyTorch model, generates a dummy input, and exports the model to ONNX format.
  """
  pytorch_model = m.DeepNeuralNet(n_x = 164, n_h = [64, 32, 8], n_y = 1)
  pytorch_model.load_state_dict(torch.load('pytorch_model.pt'))
  pytorch_model.eval()
  dummy_input = torch.tensor(np.random.randint(2, size=164), dtype=torch.float64)
  dummy_input = torch.reshape(dummy_input, (1, 164))
  print(dummy_input)
  torch.onnx.export(pytorch_model, dummy_input, 'app/public/onnx_model.onnx', verbose=True)


if __name__ == '__main__':
  main()