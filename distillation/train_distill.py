import numpy as np
import os
import sys
import time
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF

RESET = True
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Add directories to import models
sys.path.append(os.path.join(parent_dir, "v4.0w"))
sys.path.append(os.path.join(parent_dir, "v3.1"))

from model import BudgetWaveMixEMNIST
# To avoid sys.path conflicts, we import run_model directly from v3.1 using a custom import structure
import importlib.util
spec = importlib.util.spec_from_file_location("run_model_v3_1", os.path.join(parent_dir, "v3.1", "run_model.py"))
run_model_v3_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_model_v3_1)
ModelV3_1 = run_model_v3_1.ModelV3_1

# Hyperparameters
epochs = 10
batch_size = 256
learning_rate = 0.001
alpha = 0.5
temperature = 4.0

print("Loading EMNIST 28x28 dataset...")
data_path = os.path.join(parent_dir, "mnist", "emnist_28x28.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found. Please run prepare_emnist.py first.")
    sys.exit(1)

data = np.load(data_path)
raw_x_train = data['x_train'].reshape(-1, 28, 28).astype(np.float32)
raw_x_test = data['x_test'].reshape(-1, 28, 28).astype(np.float32)

# Orientation correction to upright position
x_train_corrected = np.transpose(raw_x_train, (0, 2, 1))
x_test_corrected = np.transpose(raw_x_test, (0, 2, 1))

# Channel extension: (N, 1, 28, 28)
x_train = x_train_corrected[:, np.newaxis, :, :]
x_test = x_test_corrected[:, np.newaxis, :, :]

y_train = data['y_train'].astype(np.float32)
y_test = data['y_test'].astype(np.float32)

num_train = len(x_train)
print(f"Loaded {num_train} training examples and {len(x_test)} test examples.")

if RESET:
    os.system(f"python {os.path.join(script_dir, 'reset_student.py')}")

hidden_layer_dir = os.path.join(script_dir, "hiddenLayer")

# Initialize models
student = ModelV3_1()
teacher = BudgetWaveMixEMNIST(num_classes=47)

# Load teacher model pth
teacher_path = os.path.join(parent_dir, "v4.0w", "hiddenLayer", "model.pth")
if not os.path.exists(teacher_path):
    print(f"Error: Pretrained teacher model not found at {teacher_path}.")
    sys.exit(1)

print("Loading pre-trained Teacher model...")
teacher.load_state_dict(torch.load(teacher_path, map_location="cpu", weights_only=True))

# Load student weights
from run_student import load_local_weights
if not load_local_weights(student, hidden_layer_dir):
    print("Error: Failed to load student weights.")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

student.to(device)
teacher.to(device)
teacher.eval() # Teacher is strictly frozen

# We only optimize the student's parameters
optimizer = optim.Adam(student.parameters(), lr=learning_rate)
criterion_hard = nn.CrossEntropyLoss()
criterion_soft = nn.KLDivLoss(reduction='batchmean')

def save_student_weights(model):
    np.savetxt(os.path.join(hidden_layer_dir, "conv1.txt"), model.conv1.weight.data.cpu().numpy().reshape(-1, 9), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "conv2.txt"), model.conv2.weight.data.cpu().numpy().reshape(-1, 9), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H1_trainable.txt"), model.H1_trainable.data.cpu().numpy().reshape(-1, 12), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H1_fixed.txt"), model.H1_fixed.data.cpu().numpy().reshape(-1, 12), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H2.txt"), model.H2.data.cpu().numpy(), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H3.txt"), model.H3.data.cpu().numpy(), fmt='%f')
    print("Distilled student weights saved to disk successfully.")

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

best_student_state = copy.deepcopy(student.state_dict())
best_test_acc = 0.0

print("Starting Knowledge Distillation... Press Ctrl+C to stop early.")
start_time = time.time()

try:
    for epoch in range(epochs):
        student.train()
        indices = np.random.permutation(num_train)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, num_train, batch_size):
            X_batch = torch.tensor(x_train_shuffled[i:i+batch_size]).to(device)
            Y_batch = torch.tensor(y_train_shuffled[i:i+batch_size]).to(device)
            Y_indices = torch.argmax(Y_batch, dim=1)

            # EMNIST Rotation Augmentation
            angle = random.uniform(-10.0, 10.0)
            X_batch = TF.rotate(X_batch, angle)
            
            optimizer.zero_grad()
            
            # Student logits
            student_logits = student(X_batch)
            
            # Teacher logits (frozen)
            with torch.no_grad():
                # The Teacher was trained on untransposed EMNIST images, whereas the Student expects transposed (upright) images.
                # To align domains, we transpose X_batch along axes 2 and 3 before feeding it to the Teacher.
                teacher_logits = teacher(X_batch.transpose(2, 3))
                
            # Soft targets loss (KL Divergence scaled by T^2)
            soft_loss = criterion_soft(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1)
            ) * (temperature * temperature)
            
            # Hard targets loss (standard Cross-Entropy)
            hard_loss = criterion_hard(student_logits, Y_indices)
            
            # Weighted combined loss
            loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
            loss.backward()
            
            # Check gradients
            has_nan = False
            for param in student.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan = True
                    break
                    
            if has_nan:
                print("\nGradient explosion detected. Restoring last valid model state.")
                student.load_state_dict(best_student_state)
                student.to("cpu")
                save_student_weights(student)
                raise StopIteration
                
            optimizer.step()
            
        # Evaluation step
        student.to("cpu")
        train_loss, train_acc = get_metrics(student, x_train, y_train)
        test_loss, test_acc = get_metrics(student, x_test, y_test)
        student.to(device)
        
        print(f"Epoch {epoch+1:03d}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_student_state = copy.deepcopy(student.state_dict())
            
except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted. Restoring best model...")
    student.load_state_dict(best_student_state)
    student.to("cpu")
    save_student_weights(student)
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Knowledge Distillation completed in {total_time:.2f} seconds.")

# Restore best weights and save
student.load_state_dict(best_student_state)
student.to("cpu")
print("\nSaving final distilled student weights...")
save_student_weights(student)

print("\n--- Final Test Set Evaluation ---")
final_loss, final_acc = get_metrics(student, x_test, y_test)
print(f"Distilled Student Test Accuracy: {final_acc*100:.2f}%")

# Save to benchmark CSV
import csv
csv_path = os.path.join(parent_dir, "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow(["v3.1_distilled", round(total_time, 2), round(final_loss, 4), round(final_acc*100, 2)])
print(f"Results successfully appended to {csv_path}")
