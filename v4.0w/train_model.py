import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy

from model import BudgetWaveMixEMNIST

RESET = True
device = torch.device("cuda" if torch.cuda.is_available() else "xpu")
print(f"Using device: {device}")

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading EMNIST 28x28 dataset...")
data_path = os.path.join(script_dir, "..", "mnist", "emnist_28x28.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found. Run prepare_emnist.py first.")
    sys.exit(1)

data = np.load(data_path)
# Reshape from flat (N, 784) to (N, 1, 28, 28) for PyTorch Conv2D
x_train = data['x_train'].reshape(-1, 1, 28, 28).astype(np.float32)
y_train = np.argmax(data['y_train'], axis=1).astype(np.int64)
x_test = data['x_test'].reshape(-1, 1, 28, 28).astype(np.float32)
y_test = np.argmax(data['y_test'], axis=1).astype(np.int64)

num_train = len(x_train)
print(f"Loaded {num_train} training examples and {len(x_test)} test examples.")

model_path = os.path.join(script_dir, "hiddenLayer", "model.pth")

if RESET:
    os.system(f"python {os.path.join(script_dir, 'reset_model.py')}")

model = BudgetWaveMixEMNIST(num_classes=47)
try:
    model.load_state_dict(torch.load(model_path, weights_only=True))
except FileNotFoundError:
    print("Error: Could not find model.pth. Please run reset_model.py first.")
    sys.exit(1)

model.to(device)

epochs = 30
batch_size = 256
learning_rate = 0.001

train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize dynamic plot
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
ax1, ax2, ax3, ax4 = axs.flatten()

loss_history = []
line_loss1, = ax1.plot(loss_history, label='Training Loss', color='b', linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss (Normal Scale)")
ax1.grid(True)
ax1.legend()

line_loss2, = ax2.plot(loss_history, label='Training Loss', color='b', linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_yscale("log")
ax2.set_title("Training Loss (Log Scale)")
ax2.grid(True)
ax2.legend()

time_history = []
energy_history = []
line_energy, = ax3.plot(time_history, energy_history, label='Relativistic Energy', color='r', linewidth=2)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Energy")
ax3.set_title("Model Energy vs Time")
ax3.grid(True)
ax3.legend()

train_acc_history = []
test_acc_history = []
line_train_acc, = ax4.plot(train_acc_history, label='Train Accuracy', color='g', linewidth=2)
line_test_acc, = ax4.plot(test_acc_history, label='Test Accuracy', color='orange', linewidth=2, linestyle='--')
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Accuracy")
ax4.set_title("Model Accuracy")
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.show(block=False)

def evaluate_model(data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

print("Starting training with PyTorch Adam optimizer... Press Ctrl+C to stop early.")
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
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f}")
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        test_loss, test_acc = evaluate_model(test_loader)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
        current_time = time.time() - start_time
        
        v = (test_acc - 0.1) / 0.9
        v = np.clip(v, 0, 0.9999)
        phase_angle = v * (np.pi / 2)
        energy = np.tan(phase_angle)
        
        print(f"Epoch {epoch+1:03d}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Energy: {energy:.4f}")
        
        loss_history.append(train_loss)
        line_loss1.set_ydata(loss_history)
        line_loss1.set_xdata(range(len(loss_history)))
        ax1.relim()
        ax1.autoscale_view()
        
        line_loss2.set_ydata(loss_history)
        line_loss2.set_xdata(range(len(loss_history)))
        ax2.relim()
        ax2.autoscale_view()
        
        time_history.append(current_time)
        energy_history.append(energy)
        line_energy.set_ydata(energy_history)
        line_energy.set_xdata(time_history)
        ax3.relim()
        ax3.autoscale_view()
        
        train_acc_history.append(train_acc)
        line_train_acc.set_ydata(train_acc_history)
        line_train_acc.set_xdata(range(len(train_acc_history)))
        
        test_acc_history.append(test_acc)
        line_test_acc.set_ydata(test_acc_history)
        line_test_acc.set_xdata(range(len(test_acc_history)))
        ax4.relim()
        ax4.autoscale_view()
        
        plt.pause(0.01)
        
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving best model...")
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), model_path)
    plt.ioff()
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

print("\nTraining loop finished. Saving final model...")
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), model_path)

print("\n--- Final Test Set Evaluation ---")
final_loss, final_acc = evaluate_model(test_loader)
print(f"Test Accuracy: {final_acc*100:.2f}%")

plt.ioff()
print("Close the plot window to exit the script.")
plt.savefig("v4.0_training_plot.png")

import csv
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
ver = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow([ver, round(total_time, 2), round(final_loss, 4), round(final_acc*100, 2)])
print(f"Results saved to {csv_path}")
