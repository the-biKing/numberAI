import numpy as np
import copy
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import random
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading EMNIST 28x28 dataset...")
data_path = os.path.join(script_dir, "..", "mnist", "emnist_28x28.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found.")
    sys.exit(1)

data = np.load(data_path)
raw_x_train = data['x_train'].reshape(-1, 28, 28).astype(np.float32)
raw_x_test = data['x_test'].reshape(-1, 28, 28).astype(np.float32)

# Fix the orientation by transposing the H and W axes
x_train_corrected = np.transpose(raw_x_train, (0, 2, 1))
x_test_corrected = np.transpose(raw_x_test, (0, 2, 1))

# Add the Channel dimension (C=1) for PyTorch Conv2d: (N, 1, 28, 28)
x_train = x_train_corrected[:, np.newaxis, :, :]
x_test = x_test_corrected[:, np.newaxis, :, :]

y_train = data['y_train'].astype(np.float32)
y_test = data['y_test'].astype(np.float32)

num_train = len(x_train)
print(f"Loaded {num_train} training examples and {len(x_test)} test examples.")

device = torch.device("cuda" if torch.cuda.is_available() else "xpu")

class MNISTModel300k(nn.Module):
    def __init__(self, num_classes=47):
        super(MNISTModel300k, self).__init__()
        # 參數推到約 300,000 (精確值為 304,047 參數)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256, 512)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, num_classes)

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

model = MNISTModel300k(num_classes=47).to(device)

def get_metrics(model, X, Y, batch_size=256):
    model.eval()
    losses = []
    correct = 0
    total = len(X)
    
    criterion_eval = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            X_batch = torch.tensor(X[i:i+batch_size]).to(device)
            Y_batch = torch.tensor(Y[i:i+batch_size]).to(device)
            
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
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

start_time = time.time()

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

            # Random rotation data augmentation
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
                raise StopIteration
                
            optimizer.step()
                
        # Calculate full train metrics
        train_loss, train_acc = get_metrics(model, x_train, y_train)
        test_loss, test_acc = get_metrics(model, x_test, y_test)
        
        print(f"Epoch {epoch+1:03d}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        
        best_model_state = copy.deepcopy(model.state_dict())
        
except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), os.path.join(script_dir, "emnist_model_300k.pth"))
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

model.load_state_dict(best_model_state)
print("\nTraining loop finished. Saving final model...")
torch.save(model.state_dict(), os.path.join(script_dir, "emnist_model_300k.pth"))

print("\n--- Final Test Set Evaluation ---")
final_loss, final_acc = get_metrics(model, x_test, y_test)
print(f"Test Accuracy: {final_acc*100:.2f}%")

csv_path = os.path.join(script_dir, "..", "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
ver = "example_emnist_pytorch_300k"
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow([ver, round(total_time, 2), round(final_loss, 4), round(final_acc*100, 2)])
print(f"Results saved to {csv_path}")
