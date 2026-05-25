import os
import sys
import numpy as np
import time
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Add directories to import student model from v3.1
sys.path.append(os.path.join(parent_dir, "v3.1"))
import importlib.util
spec = importlib.util.spec_from_file_location("run_model_v3_1", os.path.join(parent_dir, "v3.1", "run_model.py"))
run_model_v3_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_model_v3_1)
ModelV3_1 = run_model_v3_1.ModelV3_1

# Define directories
distillation_dir = os.path.join(parent_dir, "distillation", "hiddenLayer")
distil_q_dir = os.path.join(script_dir, "hiddenLayer")
os.makedirs(distil_q_dir, exist_ok=True)

# 1. Load trained distilled weights
print("Loading trained distilled weights from distillation/hiddenLayer...")
try:
    conv1 = np.loadtxt(os.path.join(distillation_dir, "conv1.txt")).reshape(4, 1, 3, 3)
    conv2 = np.loadtxt(os.path.join(distillation_dir, "conv2.txt")).reshape(4, 1, 3, 3)
    H1_trainable = np.loadtxt(os.path.join(distillation_dir, "H1_trainable.txt")).reshape(36, 576, 12)
    H1_fixed = np.loadtxt(os.path.join(distillation_dir, "H1_fixed.txt")).reshape(4, 26, 12)
    H2 = np.loadtxt(os.path.join(distillation_dir, "H2.txt")).reshape(480, 128)
    H3 = np.loadtxt(os.path.join(distillation_dir, "H3.txt")).reshape(128, 47)
except OSError as e:
    print(f"Error: Failed to load distilled weights: {e}")
    sys.exit(1)

weights_dict = {
    "conv1": conv1,
    "conv2": conv2,
    "H1_trainable": H1_trainable,
    "H1_fixed": H1_fixed,
    "H2": H2,
    "H3": H3
}

# 2. Symmetric INT8 Quantization Function
def quantize_tensor(W):
    max_val = np.max(np.abs(W))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    W_q = np.clip(np.round(W / scale), -128, 127).astype(np.int8)
    return W_q, scale

# 3. Quantize and save weights
print("\nQuantizing model weights to 8-bit integers (INT8)...")
quantized_weights = {}
scales = {}

for name, W in weights_dict.items():
    W_q, scale = quantize_tensor(W)
    quantized_weights[name] = W_q
    scales[name] = scale
    
    # Save quantized weights as integers (%d)
    weight_path = os.path.join(distil_q_dir, f"{name}.txt")
    np.savetxt(weight_path, W_q.reshape(-1, W_q.shape[-1]), fmt='%d')
    
    # Save scaling factor
    scale_path = os.path.join(distil_q_dir, f"{name}_scale.txt")
    np.savetxt(scale_path, np.array([scale]), fmt='%e')
    
    # Print statistics
    orig_bytes = W.nbytes
    quant_bytes = W_q.nbytes
    compression = orig_bytes / quant_bytes
    print(f"  - {name}: Float shape {W.shape} ({orig_bytes:,} bytes) -> INT8 shape {W_q.shape} ({quant_bytes:,} bytes) | Scale: {scale:.6e} | Compression: {compression:.1f}x")

print("\nQuantized INT8 weights and scales successfully saved to distil_q/hiddenLayer/.")

# 4. Dequantize weights for validation
print("\nDequantizing weights for EMNIST model evaluation...")
model = ModelV3_1()

# Dequantize back to float32
conv1_dequant = (quantized_weights["conv1"].astype(np.float32) * scales["conv1"])
conv2_dequant = (quantized_weights["conv2"].astype(np.float32) * scales["conv2"])
H1_trainable_dequant = (quantized_weights["H1_trainable"].astype(np.float32) * scales["H1_trainable"])
H1_fixed_dequant = (quantized_weights["H1_fixed"].astype(np.float32) * scales["H1_fixed"])
H2_dequant = (quantized_weights["H2"].astype(np.float32) * scales["H2"])
H3_dequant = (quantized_weights["H3"].astype(np.float32) * scales["H3"])

model.conv1.weight.data = torch.tensor(conv1_dequant, dtype=torch.float32)
model.conv2.weight.data = torch.tensor(conv2_dequant, dtype=torch.float32)
model.H1_trainable.data = torch.tensor(H1_trainable_dequant, dtype=torch.float32)
model.H1_fixed.data = torch.tensor(H1_fixed_dequant, dtype=torch.float32)
model.H2.data = torch.tensor(H2_dequant, dtype=torch.float32)
model.H3.data = torch.tensor(H3_dequant, dtype=torch.float32)

# Load EMNIST test set to verify accuracy retention
print("\nLoading EMNIST 28x28 test set...")
data_path = os.path.join(parent_dir, "mnist", "emnist_28x28.npz")
if not os.path.exists(data_path):
    data_path = os.path.join(parent_dir, "emnist_28x28.npz")

if not os.path.exists(data_path):
    print(f"Error: Dataset not found at {data_path}.")
    sys.exit(1)

data = np.load(data_path)
raw_x_test = data['x_test'].reshape(-1, 28, 28).astype(np.float32)
y_test = data['y_test'].astype(np.float32)

# Transpose for upright position
x_test_corrected = np.transpose(raw_x_test, (0, 2, 1))
x_test = x_test_corrected[:, np.newaxis, :, :]

print(f"Loaded {len(x_test)} test examples.")

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

print("Evaluating 8-bit Quantized Student on EMNIST upright test set...")
start_eval_time = time.time()
test_loss, test_acc = evaluate_model(model, x_test, y_test)
eval_duration = time.time() - start_eval_time

print(f"\n--- Evaluation Results ---")
print(f"Quantized Model Test Loss: {test_loss:.4f}")
print(f"Quantized Model Test Accuracy: {test_acc*100:.2f}%")
print(f"Evaluation took {eval_duration:.2f} seconds.")

# Save to benchmark CSV
import csv
csv_path = os.path.join(parent_dir, "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    # Quantization does not have training time, so we log it as 0.0 or N/A
    writer.writerow(["v3.1_distilled_quantized", 0.00, round(test_loss, 4), round(test_acc*100, 2)])
print(f"\nResults successfully appended to {csv_path}")
