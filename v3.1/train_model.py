import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import time

RESET = True
script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading EMNIST 28x28 dataset...")
data_path = os.path.join(script_dir, "..", "mnist", "emnist_28x28.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found. Run prepare_emnist.py first.")
    sys.exit(1)

data = np.load(data_path)
# PyTorch expects shape (N, C, H, W) for images
x_train = data['x_train'].reshape(-1, 1, 28, 28).astype(np.float32)
y_train = data['y_train'].astype(np.float32)
x_test = data['x_test'].reshape(-1, 1, 28, 28).astype(np.float32)
y_test = data['y_test'].astype(np.float32)

num_train = len(x_train)
print(f"Loaded {num_train} training examples and {len(x_test)} test examples.")

if RESET:
    os.system(f"python {os.path.join(script_dir, 'reset_model.py')}")

sys.path.append(script_dir)
from run_model import ModelV3_1, load_weights

model = ModelV3_1()
hidden_layer_dir = os.path.join(script_dir, "hiddenLayer")

if not load_weights(model, hidden_layer_dir):
    print("Error: Could not find model weights. Please run reset_model.py first.")
    sys.exit(1)

def save_model(model):
    np.savetxt(os.path.join(hidden_layer_dir, "conv1.txt"), model.conv1.weight.data.numpy().reshape(-1, 9), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "conv2.txt"), model.conv2.weight.data.numpy().reshape(-1, 9), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H1_1.txt"), model.H1_1.data.numpy().reshape(-1, 24), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H1_2.txt"), model.H1_2.data.numpy().reshape(-1, 12), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H1_3.txt"), model.H1_3.data.numpy().reshape(-1, 8), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H1_4.txt"), model.H1_4.data.numpy().reshape(-1, 4), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H1_5.txt"), model.H1_5.data.numpy().reshape(-1, 3), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H1_6.txt"), model.H1_6.data.numpy().reshape(-1, 2), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H1_7.txt"), model.H1_7.data.numpy().reshape(-1, 1), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H2.txt"), model.H2.data.numpy(), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H3.txt"), model.H3.data.numpy(), fmt='%f')
    print("Model saved to disk successfully.")

def get_metrics(model, X, Y, batch_size=256):
    model.eval()
    losses = []
    correct = 0
    total = len(X)
    
    criterion_eval = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            X_batch = torch.tensor(X[i:i+batch_size])
            Y_batch = torch.tensor(Y[i:i+batch_size])
            
            A = model(X_batch)
            truths = torch.argmax(Y_batch, dim=1)
            loss = criterion_eval(A, truths).item()
            losses.append(loss * len(X_batch))
            
            preds = torch.argmax(A, dim=1)
            correct += (preds == truths).sum().item()
            
    return sum(losses) / total, correct / total

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

best_model_state = copy.deepcopy(model.state_dict())

print("Starting PyTorch training... Press Ctrl+C to stop early.")

epochs = 100
batch_size = 256
learning_rate = 0.001

lr_conv1 = learning_rate
lr_conv2 = learning_rate
lr_H1 = learning_rate
lr_H2 = learning_rate
lr_H3 = learning_rate

optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': lr_conv1},
    {'params': model.conv2.parameters(), 'lr': lr_conv2},
    {'params': [model.H1_1, model.H1_2, model.H1_3, model.H1_4, model.H1_5, model.H1_6, model.H1_7], 'lr': lr_H1},
    {'params': [model.H2], 'lr': lr_H2},
    {'params': [model.H3], 'lr': lr_H3}
], lr=learning_rate)

criterion = nn.CrossEntropyLoss()

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

try:
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(num_train)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, num_train, batch_size):
            X_batch = torch.tensor(x_train_shuffled[i:i+batch_size]).to(device)
            Y_batch = torch.tensor(y_train_shuffled[i:i+batch_size]).to(device)
            Y_indices = torch.argmax(Y_batch, dim=1)
            
            optimizer.zero_grad()
            A = model(X_batch)
            loss = criterion(A, Y_indices)
            loss.backward()
            
            # Check for exploding gradients
            has_nan = False
            for param in model.parameters():
                if torch.isnan(param.grad).any():
                    has_nan = True
                    break
            
            if has_nan:
                print(f"\nGradient explosion detected. Restoring last valid model.")
                model.load_state_dict(best_model_state)
                model.to("cpu")
                save_model(model)
                raise StopIteration
                
            optimizer.step()
                
        # Calculate full train metrics
        model.to("cpu")
        train_loss, train_acc = get_metrics(model, x_train, y_train)
        test_loss, test_acc = get_metrics(model, x_test, y_test)
        model.to(device)
        
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
        
        best_model_state = copy.deepcopy(model.state_dict())
        
except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    model.load_state_dict(best_model_state)
    model.to("cpu")
    save_model(model)
    plt.ioff()
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

model.load_state_dict(best_model_state)
model.to("cpu")

print("\nTraining loop finished. Saving final model...")
save_model(model)

print("\n--- Final Test Set Evaluation ---")
final_loss, final_acc = get_metrics(model, x_test, y_test)
print(f"Test Accuracy: {final_acc*100:.2f}%")

plt.ioff()
print("Close the plot window to exit the script.")
plt.savefig("v3.1_training_plot.png")

import csv
csv_path = os.path.join(script_dir, "..", "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
ver = os.path.basename(script_dir)
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow([ver, round(total_time, 2), round(final_loss, 4), round(final_acc*100, 2)])
print(f"Results saved to {csv_path}")
