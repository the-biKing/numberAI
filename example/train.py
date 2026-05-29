import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading MNIST 16x16 dataset...")
data_path = os.path.join(script_dir, "..", "mnist", "mnist_16x16.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found.")
    sys.exit(1)

data = np.load(data_path)
# Reshape to (N, C, H, W) for PyTorch Conv2d (16x16 images)
x_train = data['x_train'].reshape(-1, 1, 16, 16)
y_train = data['y_train']
x_test = data['x_test'].reshape(-1, 1, 16, 16)
y_test = data['y_test']

num_train = len(x_train)
print(f"Loaded {num_train} training examples and {len(x_test)} test examples.")

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
y_train_idx = torch.argmax(y_train_t, dim=1)

x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
y_test_idx = torch.argmax(y_test_t, dim=1)

# Define the MPU6050 model architecture adapted for 2D images
class MNISTModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        return x

model = MNISTModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def get_metrics(A_vals, expected):
    # Mean Euclidean loss
    loss = np.mean(np.sqrt(np.sum((A_vals - expected)**2, axis=1)))
    # Accuracy
    preds = np.argmax(A_vals, axis=1)
    truths = np.argmax(expected, axis=1)
    acc = np.mean(preds == truths)
    return loss, acc

# Initialize dynamic plot
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
ax1, ax2, ax3, ax4 = axs.flatten()

# Subplot 1: Loss (Normal Scale)
loss_history = []
line_loss1, = ax1.plot(loss_history, label='Training Loss', color='b', linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss (Normal Scale)")
ax1.grid(True)
ax1.legend()

# Subplot 2: Loss (Log Scale)
line_loss2, = ax2.plot(loss_history, label='Training Loss', color='b', linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_yscale("log")
ax2.set_title("Training Loss (Log Scale)")
ax2.grid(True)
ax2.legend()

# Subplot 3: Energy vs Time
time_history = []
energy_history = []
line_energy, = ax3.plot(time_history, energy_history, label='Relativistic Energy', color='r', linewidth=2)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Energy")
ax3.set_title("Model Energy vs Time")
ax3.grid(True)
ax3.legend()

# Subplot 4: Accuracy
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

best_state_dict = copy.deepcopy(model.state_dict())

print("Starting training with PyTorch... Press Ctrl+C to stop early.")

epochs = 100
batch_size = 256
start_time = time.time()

try:
    for epoch in range(epochs):
        model.train()
        # Shuffle dataset each epoch
        indices = torch.randperm(num_train)
        x_train_shuffled = x_train_t[indices]
        y_train_idx_shuffled = y_train_idx[indices]
        
        for i in range(0, num_train, batch_size):
            X_batch = x_train_shuffled[i:i+batch_size]
            Y_batch = y_train_idx_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
                
        # Calculate full train metrics for epoch logging
        model.eval()
        with torch.no_grad():
            # For large dataset, process in chunks if needed, but 16x16 MNIST is small enough
            train_A = model(x_train_t).cpu().numpy()
            train_loss, train_acc = get_metrics(train_A, y_train)
            
            test_A = model(x_test_t).cpu().numpy()
            test_loss, test_acc = get_metrics(test_A, y_test)
        
        current_time = time.time() - start_time
        
        # 1. Linearly stretch 10%-100% to 0.0-1.0
        v = (test_acc - 0.1) / 0.9
        v = np.clip(v, 0, 0.9999)

        # 2. Map 0-1 to the 0 to pi/2 range
        phase_angle = v * (np.pi / 2)

        # 3. Calculate Energy using Tangent
        energy = np.tan(phase_angle)
        
        print(f"Epoch {epoch+1:03d}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Energy: {energy:.4f}")
        
        # Update plots
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
        
        # Epoch completed successfully
        best_state_dict = copy.deepcopy(model.state_dict())
        
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), os.path.join(script_dir, "example_model.pth"))
    plt.ioff()
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

print("\nTraining loop finished. Saving final model...")
model.load_state_dict(best_state_dict)
torch.save(model.state_dict(), os.path.join(script_dir, "example_model.pth"))

print("\n--- Final Test Set Evaluation ---")
model.eval()
with torch.no_grad():
    test_A = model(x_test_t).cpu().numpy()
final_loss, final_acc = get_metrics(test_A, y_test)
print(f"Test Accuracy: {final_acc*100:.2f}%")

plt.ioff()
print("Close the plot window to exit the script.")
plt.savefig(os.path.join(script_dir, "example_training_plot.png"))

import csv
csv_path = os.path.join(script_dir, "..", "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
ver = "example_pytorch"
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow([ver, round(total_time, 2), round(final_loss, 4), round(final_acc*100, 2)])
print(f"Results saved to {csv_path}")
