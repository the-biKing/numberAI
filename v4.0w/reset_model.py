import os
import torch
from model import BudgetWaveMixEMNIST

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "hiddenLayer")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Initialize the new WaveMix model
model = BudgetWaveMixEMNIST()

# Save the PyTorch state_dict
model_path = os.path.join(target_dir, "model.pth")
torch.save(model.state_dict(), model_path)

print(f"Weights reset successfully in: {model_path}")
