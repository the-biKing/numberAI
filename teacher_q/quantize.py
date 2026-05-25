import os
import sys
import numpy as np
import time
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Add directories to import teacher model
sys.path.append(os.path.join(parent_dir, "v4.0w"))
from model import BudgetWaveMixEMNIST

# Define directories
teacher_dir = os.path.join(parent_dir, "v4.0w", "hiddenLayer")
teacher_q_dir = os.path.join(script_dir, "hiddenLayer")
os.makedirs(teacher_q_dir, exist_ok=True)

# 1. Load pre-trained teacher model
print("Loading pre-trained Teacher model from v4.0w/hiddenLayer...")
model = BudgetWaveMixEMNIST(num_classes=47)
model_path = os.path.join(teacher_dir, "model.pth")
if not os.path.exists(model_path):
    print(f"Error: Pretrained teacher model not found at {model_path}.")
    sys.exit(1)

model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model.eval()

# 2. Symmetric INT4 Quantization
print("\nQuantizing teacher parameters to 4-bit (INT4)...")
quantized_state_dict = {}
scales = {}

# We only quantize floating point trainable parameters (nn.Parameter)
# Non-trainable registered buffers like WaveMix's Haar wavelet kernels are left untouched
total_original_bytes = 0
total_quantized_bytes = 0

for name, param in model.named_parameters():
    param_data = param.data.numpy()
    orig_bytes = param_data.nbytes
    total_original_bytes += orig_bytes
    
    # Symmetric 4-bit quantization maps to range [-7, 7]
    max_val = np.max(np.abs(param_data))
    scale = max_val / 7.0 if max_val > 0 else 1.0
    
    # Quantize and clamp to [-7, 7]
    W_q = np.clip(np.round(param_data / scale), -7, 7).astype(np.int8)
    
    quantized_state_dict[name] = torch.tensor(W_q, dtype=torch.int8)
    scales[name] = scale
    
    # Calculate quantized bytes (4 bits per parameter = 0.5 bytes)
    quant_bytes = int(np.ceil(param_data.size * 0.5))
    total_quantized_bytes += quant_bytes
    
    print(f"  - {name}: Float shape {param_data.shape} ({orig_bytes:,} B) -> INT4 ({quant_bytes:,} B) | Scale: {scale:.6e}")

# Copy running batchnorm buffers (running_mean, running_var, etc.) to quantized checkpoint without quantization
checkpoint = {
    "quantized_state_dict": quantized_state_dict,
    "scales": scales,
    "buffers": {name: buffer.clone() for name, buffer in model.named_buffers()}
}

checkpoint_path = os.path.join(teacher_q_dir, "model_q4.pth")
torch.save(checkpoint, checkpoint_path)

print(f"\nModel quantization to INT4 complete!")
print(f"  - Trainable parameters float32 size: {total_original_bytes / (1024*1024):.2f} MB ({total_original_bytes:,} bytes)")
print(f"  - Trainable parameters quantized INT4 size: {total_quantized_bytes / 1024:.2f} KB ({total_quantized_bytes:,} bytes)")
print(f"  - Theoretical compression factor: {total_original_bytes / total_quantized_bytes:.1f}x")
print(f"Quantized model checkpoint successfully saved to {checkpoint_path}")

# 3. Dequantize weights for validation
print("\nDequantizing weights for EMNIST teacher evaluation...")
for name, param in model.named_parameters():
    W_q = quantized_state_dict[name].numpy().astype(np.float32)
    scale = scales[name]
    dequantized = W_q * scale
    param.data = torch.tensor(dequantized, dtype=torch.float32)

# Load EMNIST test set
# Note: The teacher model expects raw EMNIST test set without transposing to upright position!
print("\nLoading raw EMNIST 28x28 test set...")
data_path = os.path.join(parent_dir, "mnist", "emnist_28x28.npz")
if not os.path.exists(data_path):
    data_path = os.path.join(parent_dir, "emnist_28x28.npz")

if not os.path.exists(data_path):
    print(f"Error: Dataset not found at {data_path}.")
    sys.exit(1)

data = np.load(data_path)
raw_x_test = data['x_test'].reshape(-1, 28, 28).astype(np.float32)
y_test = data['y_test'].astype(np.float32)

# Extend channels to (N, 1, 28, 28)
x_test = raw_x_test[:, np.newaxis, :, :]
print(f"Loaded {len(x_test)} test examples in raw EMNIST orientation.")

# Evaluation function
def evaluate_model(model, X, Y, batch_size=256):
    model.eval()
    correct = 0
    total = len(X)
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            X_batch = torch.tensor(X[i:i+batch_size]).to(device)
            Y_batch = torch.tensor(Y[i:i+batch_size]).to(device)
            
            A = model(X_batch)
            truths = torch.argmax(Y_batch, dim=1)
            loss = criterion(A, truths).item()
            losses.append(loss * len(X_batch))
            
            preds = torch.argmax(A, dim=1)
            correct += (preds == truths).sum().item()
            
    return sum(losses) / total, correct / total

print("Evaluating 4-bit Quantized Teacher on EMNIST raw test set...")
start_eval_time = time.time()
test_loss, test_acc = evaluate_model(model, x_test, y_test)
eval_duration = time.time() - start_eval_time

print(f"\n--- Evaluation Results ---")
print(f"INT4 Teacher Test Loss: {test_loss:.4f}")
print(f"INT4 Teacher Test Accuracy: {test_acc*100:.2f}%")
print(f"Evaluation took {eval_duration:.2f} seconds.")

# Save to benchmark CSV
import csv
csv_path = os.path.join(parent_dir, "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow(["v4.0w_quantized_int4", 0.00, round(test_loss, 4), round(test_acc*100, 2)])
print(f"\nResults successfully appended to {csv_path}")
