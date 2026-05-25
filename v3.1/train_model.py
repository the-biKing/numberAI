import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
import sys
import time
import torchvision.transforms.functional as TF
import random

# Fixed 3x3 Gaussian Blur kernel for Step 1 (conv1)
GAUSSIAN_BLUR = np.array([
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
], dtype=np.float32)

# Fixed 3x3 kernels for Step 2 (conv2)
SOBEL_X = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.float32)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

SOBEL_DIAG1 = np.array([
    [-2, -1,  0],
    [-1,  0,  1],
    [ 0,  1,  2]
], dtype=np.float32)

SOBEL_DIAG2 = np.array([
    [ 0, -1, -2],
    [ 1,  0, -1],
    [ 2,  1,  0]
], dtype=np.float32)

RESET = True
script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading EMNIST 28x28 dataset...")
data_path = os.path.join(script_dir, "..", "mnist", "emnist_28x28.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found. Run prepare_emnist.py first.")
    sys.exit(1)

data = np.load(data_path)

# 1. Load the raw arrays
raw_x_train = data['x_train'].reshape(-1, 28, 28).astype(np.float32)
raw_x_test = data['x_test'].reshape(-1, 28, 28).astype(np.float32)

# 2. Fix the orientation by transposing the H and W axes (axes 1 and 2)
# This flips the images back to their natural upright position
x_train_corrected = np.transpose(raw_x_train, (0, 2, 1))
x_test_corrected = np.transpose(raw_x_test, (0, 2, 1))

# 3. Add the Channel dimension (C=1) that PyTorch expects: (N, 1, 28, 28)
x_train = x_train_corrected[:, np.newaxis, :, :]
x_test = x_test_corrected[:, np.newaxis, :, :]

y_train = data['y_train'].astype(np.float32)
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
    np.savetxt(os.path.join(hidden_layer_dir, "H1_trainable.txt"), model.H1_trainable.data.numpy().reshape(-1, 12), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H1_fixed.txt"), model.H1_fixed.data.numpy().reshape(-1, 12), fmt='%f')
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

best_model_state = copy.deepcopy(model.state_dict())

print("Starting PyTorch training... Press Ctrl+C to stop early.")

epochs = 10
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
    {'params': [model.H1_trainable], 'lr': lr_H1},
    {'params': [model.H1_fixed], 'lr': lr_H1},
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

            angle = random.uniform(-10.0, 10.0)
            X_batch = TF.rotate(X_batch, angle)
            
            optimizer.zero_grad()
            A = model(X_batch)
            loss = criterion(A, Y_indices)
            loss.backward()
            

            
            # Check for exploding gradients
            has_nan = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
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
        
        print(f"Epoch {epoch+1:03d}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        
        best_model_state = copy.deepcopy(model.state_dict())
        
except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    model.load_state_dict(best_model_state)
    model.to("cpu")
    save_model(model)
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
