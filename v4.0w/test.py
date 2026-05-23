import torch
import sys
import os
from model import BudgetWaveMixEMNIST

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

print("Testing BudgetWaveMixEMNIST forward and backward pass...")

# Dummy input: (Batch_size, Channels, Height, Width)
N = 2
X = torch.randn(N, 1, 28, 28)
Y = torch.randint(0, 47, (N,))

# Initialize model
model = BudgetWaveMixEMNIST(num_classes=47)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Forward pass
outputs = model(X)
print(f"Output shape: {outputs.shape}") # Should be (2, 47)

# Backward pass
loss = criterion(outputs, Y)
optimizer.zero_grad()
loss.backward()

# Check gradients
has_grad = False
for name, param in model.named_parameters():
    if param.grad is not None:
        has_grad = True
        break

if has_grad:
    print("Success! Gradients computed successfully.")
else:
    print("Error: Gradients were not computed.")
    
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")
