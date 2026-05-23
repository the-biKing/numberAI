import numpy as np
import sys
import os
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from run_model import ModelV3_1

N = 2
# X is an image batch in PyTorch: (N, C, H, W)
X = torch.randn(N, 1, 28, 28, dtype=torch.float32)
Y = torch.randn(N, 47, dtype=torch.float32)

model = ModelV3_1()

# Forward pass
out = model(X)
print(f"Forward pass output shape: {out.shape}")

# Backward pass
loss = torch.mean(torch.sqrt(torch.sum((out - Y)**2, dim=1)))
loss.backward()

print("Success! Gradients computed via PyTorch.")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"Gradient for {name} computed. Shape: {param.grad.shape}")
