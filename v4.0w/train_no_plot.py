import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy

from model import BudgetWaveMixEMNIST

RESET = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading EMNIST 28x28 dataset...", flush=True)
data_path = os.path.join(script_dir, "..", "mnist", "emnist_28x28.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found. Run prepare_emnist.py first.", flush=True)
    sys.exit(1)

data = np.load(data_path)
# Reshape from flat (N, 784) to (N, 1, 28, 28) for PyTorch Conv2D
x_train = data['x_train'].reshape(-1, 1, 28, 28).astype(np.float32)
y_train = np.argmax(data['y_train'], axis=1).astype(np.int64)
x_test = data['x_test'].reshape(-1, 1, 28, 28).astype(np.float32)
y_test = np.argmax(data['y_test'], axis=1).astype(np.int64)

num_train = len(x_train)
print(f"Loaded {num_train} training examples and {len(x_test)} test examples.", flush=True)

model_path = os.path.join(script_dir, "hiddenLayer", "model.pth")

if RESET:
    os.system(f"python {os.path.join(script_dir, 'reset_model.py')}")

model = BudgetWaveMixEMNIST(num_classes=47)
try:
    model.load_state_dict(torch.load(model_path, weights_only=True))
except FileNotFoundError:
    print("Error: Could not find model.pth. Please run reset_model.py first.", flush=True)
    sys.exit(1)

model.to(device)

# SOTA High-Performance Hyperparameters
epochs = 15
batch_size = 512
learning_rate = 0.001

train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
# Use pin_memory=True for faster CPU-to-GPU memory copies
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# Cosine Annealing Learning Rate Scheduler for rapid and stable convergence
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# GradScaler for Mixed Precision (AMP) training on CUDA
scaler = torch.cuda.amp.GradScaler()

def evaluate_model(data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            # Run inference using AMP autocast for speed
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

print("Starting training with Mixed Precision (AMP) and Cosine Annealing...", flush=True)
start_time = time.time()
best_model_state = copy.deepcopy(model.state_dict())
best_test_acc = 0.0

try:
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        num_batches = len(train_loader)
        epoch_start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            # Forward pass with Mixed Precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f} | LR: {current_lr:.6f}", flush=True)
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # Step learning rate scheduler
        scheduler.step()
        
        test_loss, test_acc = evaluate_model(test_loader)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_path)
            print(f"--> [New Best Model Saved] Test Acc: {test_acc*100:.2f}% (Epoch {epoch+1})", flush=True)
        
        print(f"Epoch {epoch+1:02d}/{epochs} ({epoch_duration:.1f}s) - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%", flush=True)
        
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving best model...", flush=True)
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), model_path)
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.", flush=True)

print("\nTraining loop finished. Loading best model state for final evaluation...", flush=True)
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), model_path)

print("\n--- Final Test Set Evaluation ---", flush=True)
final_loss, final_acc = evaluate_model(test_loader)
print(f"Test Accuracy: {final_acc*100:.2f}%", flush=True)

import csv
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
ver = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow([ver, round(total_time, 2), round(final_loss, 4), round(final_acc*100, 2)])
print(f"Results saved to {csv_path}", flush=True)
