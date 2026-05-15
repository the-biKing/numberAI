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
x_train = data['x_train'].reshape(data['x_train'].shape[0], -1)
y_train = data['y_train']
x_test = data['x_test'].reshape(data['x_test'].shape[0], -1)
y_test = data['y_test']

num_train = len(x_train)
print(f"Loaded {num_train} training examples and {len(x_test)} test examples.")

H1_1_path = os.path.join(script_dir, "hiddenLayer", "H1_1.txt")
H1_2_path = os.path.join(script_dir, "hiddenLayer", "H1_2.txt")
H1_3_path = os.path.join(script_dir, "hiddenLayer", "H1_3.txt")
H1_4_path = os.path.join(script_dir, "hiddenLayer", "H1_4.txt")
H1_5_path = os.path.join(script_dir, "hiddenLayer", "H1_5.txt")
H2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")
H_conv_path = os.path.join(script_dir, "hiddenLayer", "H_conv.txt")
H3_path = os.path.join(script_dir, "hiddenLayer", "H3.txt")

if RESET:
    os.system(f"python {os.path.join(script_dir, 'reset_model.py')}")

try:
    H1_1 = np.loadtxt(H1_1_path)
    H1_2 = np.loadtxt(H1_2_path)
    H1_3 = np.loadtxt(H1_3_path)
    H1_4 = np.loadtxt(H1_4_path)
    H1_5 = np.loadtxt(H1_5_path)
    H2 = np.loadtxt(H2_path)
    H_conv = np.loadtxt(H_conv_path).reshape(3, 3)
    H3 = np.loadtxt(H3_path)
except OSError:
    print("Error: Could not find model weights. Please run reset_model.py first.")
    sys.exit(1)

def forward_pass(X, H1_1_mat, H1_2_mat, H1_3_mat, H1_4_mat, H1_5_mat, H2_mat, H_conv_mat, H3_mat):
    N = X.shape[0]
    
    O1 = np.dot(X, H1_1_mat)
    
    X2 = X.reshape(N, 2, 392)
    O2 = np.dot(X2, H1_2_mat)
    
    X3 = X.reshape(N, 7, 112)
    O3 = np.dot(X3, H1_3_mat)
    
    X4 = X.reshape(N, 14, 56)
    O4 = np.dot(X4, H1_4_mat)
    
    X5 = X.reshape(N, 28, 28)
    O5 = np.dot(X5, H1_5_mat)
    
    O1_flat = O1.reshape(N, 56)
    O2_flat = O2.reshape(N, 56)
    O3_flat = O3.reshape(N, 56)
    O4_flat = O4.reshape(N, 56)
    O5_flat = O5.reshape(N, 56)
    
    concat_O = np.concatenate((O1_flat, O2_flat, O3_flat, O4_flat, O5_flat), axis=1)
    M1 = np.maximum(0, concat_O)
    
    M2_branch1 = np.dot(M1, H2_mat)
    
    X_img = X.reshape(N, 28, 28)
    X_pad = np.pad(X_img, ((0,0), (1,1), (1,1)), mode='constant')
    shape = (N, 14, 14, 3, 3)
    strides = (X_pad.strides[0], X_pad.strides[1]*2, X_pad.strides[2]*2, X_pad.strides[1], X_pad.strides[2])
    X_blocks = np.lib.stride_tricks.as_strided(X_pad, shape=shape, strides=strides)
    
    M2_branch2 = np.tensordot(X_blocks, H_conv_mat, axes=([3, 4], [0, 1]))
    M2_branch2 = M2_branch2.reshape(N, 196)
    
    M2 = M2_branch1 + M2_branch2
    M2_relu = np.maximum(0, M2)
    
    A = np.dot(M2_relu, H3_mat)
    return O1, O2, O3, O4, O5, M1, X_blocks, M2, M2_relu, A

def get_metrics(A_vals, expected):
    loss = np.mean(np.sqrt(np.sum((A_vals - expected)**2, axis=1)))
    preds = np.argmax(A_vals, axis=1)
    truths = np.argmax(expected, axis=1)
    acc = np.mean(preds == truths)
    return loss, acc

def save_model(H1_1_mat, H1_2_mat, H1_3_mat, H1_4_mat, H1_5_mat, H2_mat, H_conv_mat, H3_mat):
    np.savetxt(H1_1_path, H1_1_mat, fmt='%f')
    np.savetxt(H1_2_path, H1_2_mat, fmt='%f')
    np.savetxt(H1_3_path, H1_3_mat, fmt='%f')
    np.savetxt(H1_4_path, H1_4_mat, fmt='%f')
    np.savetxt(H1_5_path, H1_5_mat, fmt='%f')
    np.savetxt(H2_path, H2_mat, fmt='%f')
    np.savetxt(H_conv_path, H_conv_mat, fmt='%f')
    np.savetxt(H3_path, H3_mat, fmt='%f')
    print("Model saved to disk successfully.")

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

best_H1_1 = copy.deepcopy(H1_1)
best_H1_2 = copy.deepcopy(H1_2)
best_H1_3 = copy.deepcopy(H1_3)
best_H1_4 = copy.deepcopy(H1_4)
best_H1_5 = copy.deepcopy(H1_5)
best_H2 = copy.deepcopy(H2)
best_H_conv = copy.deepcopy(H_conv)
best_H3 = copy.deepcopy(H3)

print("Starting training with Mini-batch Backpropagation... Press Ctrl+C to stop early.")

epochs = 100
batch_size = 256
LR_H1 = 0.05
LR_H2 = 0.05
LR_H_conv = 0.05
LR_H3 = 0.05

start_time = time.time()

try:
    for epoch in range(epochs):
        indices = np.random.permutation(num_train)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, num_train, batch_size):
            X_batch = x_train_shuffled[i:i+batch_size]
            Y_batch = y_train_shuffled[i:i+batch_size]
            batch_n = len(X_batch)
            
            # Forward Pass
            O1, O2, O3, O4, O5, M1, X_blocks, M2, M2_relu, A = forward_pass(
                X_batch, H1_1, H1_2, H1_3, H1_4, H1_5, H2, H_conv, H3)
            
            # Backward Pass
            error_vector = (A - Y_batch) / batch_n
            
            dH3 = np.dot(M2_relu.T, error_vector)
            
            hidden_error_m2 = np.dot(error_vector, H3.T)
            relu_deriv_m2 = (M2 > 0).astype(float)
            hidden_error_pre_relu_m2 = hidden_error_m2 * relu_deriv_m2
            
            dH2 = np.dot(M1.T, hidden_error_pre_relu_m2)
            
            hidden_error_m2_spatial = hidden_error_pre_relu_m2.reshape(batch_n, 14, 14)
            dH_conv = np.tensordot(hidden_error_m2_spatial, X_blocks, axes=([0, 1, 2], [0, 1, 2]))
            
            hidden_error_m1 = np.dot(hidden_error_pre_relu_m2, H2.T)
            relu_deriv_m1 = (M1 > 0).astype(float)
            hidden_error_pre_relu_m1 = hidden_error_m1 * relu_deriv_m1
            
            dO1_flat = hidden_error_pre_relu_m1[:, 0:56]
            dO2_flat = hidden_error_pre_relu_m1[:, 56:112]
            dO3_flat = hidden_error_pre_relu_m1[:, 112:168]
            dO4_flat = hidden_error_pre_relu_m1[:, 168:224]
            dO5_flat = hidden_error_pre_relu_m1[:, 224:280]
            
            dO1 = dO1_flat.reshape(O1.shape)
            dO2 = dO2_flat.reshape(O2.shape)
            dO3 = dO3_flat.reshape(O3.shape)
            dO4 = dO4_flat.reshape(O4.shape)
            dO5 = dO5_flat.reshape(O5.shape)
            
            dH1_1 = np.dot(X_batch.T, dO1)
            
            X2 = X_batch.reshape(batch_n, 2, 392)
            dH1_2 = np.tensordot(X2, dO2, axes=([0, 1], [0, 1]))
            
            X3 = X_batch.reshape(batch_n, 7, 112)
            dH1_3 = np.tensordot(X3, dO3, axes=([0, 1], [0, 1]))
            
            X4 = X_batch.reshape(batch_n, 14, 56)
            dH1_4 = np.tensordot(X4, dO4, axes=([0, 1], [0, 1]))
            
            X5 = X_batch.reshape(batch_n, 28, 28)
            dH1_5 = np.tensordot(X5, dO5, axes=([0, 1], [0, 1]))
            
            # Apply Gradients
            H1_1 -= LR_H1 * dH1_1
            H1_2 -= LR_H1 * dH1_2
            H1_3 -= LR_H1 * dH1_3
            H1_4 -= LR_H1 * dH1_4
            H1_5 -= LR_H1 * dH1_5
            H2 -= LR_H2 * dH2
            H_conv -= LR_H_conv * dH_conv
            H3 -= LR_H3 * dH3
            
            if np.isnan(np.sum(H3)):
                print(f"\nGradient explosion detected. Restoring last valid model.")
                save_model(best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H1_5, best_H2, best_H_conv, best_H3)
                raise StopIteration
                
        # Calculate full train metrics for epoch logging
        _, _, _, _, _, _, _, _, _, train_A = forward_pass(x_train, H1_1, H1_2, H1_3, H1_4, H1_5, H2, H_conv, H3)
        train_loss, train_acc = get_metrics(train_A, y_train)
        
        _, _, _, _, _, _, _, _, _, test_A = forward_pass(x_test, H1_1, H1_2, H1_3, H1_4, H1_5, H2, H_conv, H3)
        test_loss, test_acc = get_metrics(test_A, y_test)
        
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
        
        best_H1_1 = copy.deepcopy(H1_1)
        best_H1_2 = copy.deepcopy(H1_2)
        best_H1_3 = copy.deepcopy(H1_3)
        best_H1_4 = copy.deepcopy(H1_4)
        best_H1_5 = copy.deepcopy(H1_5)
        best_H2 = copy.deepcopy(H2)
        best_H_conv = copy.deepcopy(H_conv)
        best_H3 = copy.deepcopy(H3)
        
except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    save_model(best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H1_5, best_H2, best_H_conv, best_H3)
    plt.ioff()
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

print("\nTraining loop finished. Saving final model...")
save_model(best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H1_5, best_H2, best_H_conv, best_H3)

print("\n--- Final Test Set Evaluation ---")
_, _, _, _, _, _, _, _, _, test_A = forward_pass(x_test, best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H1_5, best_H2, best_H_conv, best_H3)
final_loss, final_acc = get_metrics(test_A, y_test)
print(f"Test Accuracy: {final_acc*100:.2f}%")

plt.ioff()
print("Close the plot window to exit the script.")
plt.savefig("v3.0_training_plot.png")

import csv
import time
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
ver = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow([ver, round(total_time, 2), round(final_loss, 4), round(final_acc*100, 2)])
print(f"Results saved to {csv_path}")
