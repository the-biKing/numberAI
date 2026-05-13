import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import time

RESET = True

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading MNIST 16x16 dataset...")
data_path = os.path.join(script_dir, "..", "mnist", "mnist_16x16.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found. Run prepare_mnist.py first.")
    sys.exit(1)

data = np.load(data_path)
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

num_train = len(x_train)
print(f"Loaded {num_train} training examples and {len(x_test)} test examples.")

H1_path = os.path.join(script_dir, "hiddenLayer", "H1.txt")
H2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")

if RESET:
    os.system(f"python {os.path.join(script_dir, 'reset_model.py')}")

try:
    H1 = np.loadtxt(H1_path)
    H2 = np.loadtxt(H2_path)
except OSError:
    print("Error: Could not find H1.txt or H2.txt. Please run reset_model.py first.")
    sys.exit(1)

def forward_pass(X, H1_mat, H2_mat):
    # Linear projection through H1 (36 neurons)
    M1_linear = np.dot(X, H1_mat)
    
    # Max pooling 2x2 to the 16x16 input -> 8x8
    x_16x16 = X.reshape(-1, 16, 16)
    pool1 = x_16x16.reshape(-1, 8, 2, 8, 2).max(axis=(2, 4))
    
    # 3x3 Convolution with fixed Laplacian kernel (no padding) -> 6x6
    laplacian_kernel = np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float32)
    
    conv_out = np.zeros((X.shape[0], 6, 6), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            conv_out += pool1[:, i:i+6, j:j+6] * laplacian_kernel[i, j]
            
    # Flatten -> 36 features
    pool_conv_flat = conv_out.reshape(-1, 36)
    
    # ResNet Addition: add extracted features to linear projection
    M_pre_relu = M1_linear + pool_conv_flat
    
    # ReLU activation
    M = np.maximum(0, M_pre_relu)
    
    A = np.dot(M, H2_mat)
    return M, A

def get_metrics(A_vals, expected):
    # Mean Euclidean loss
    loss = np.mean(np.sqrt(np.sum((A_vals - expected)**2, axis=1)))
    
    # Accuracy
    preds = np.argmax(A_vals, axis=1)
    truths = np.argmax(expected, axis=1)
    acc = np.mean(preds == truths)
    return loss, acc

def save_model(H1_mat, H2_mat):
    np.savetxt(H1_path, H1_mat, fmt='%f')
    np.savetxt(H2_path, H2_mat, fmt='%f')
    print("Model saved to disk successfully.")

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

best_H1 = copy.deepcopy(H1)
best_H2 = copy.deepcopy(H2)

print("Starting training with Mini-batch Backpropagation... Press Ctrl+C to stop early.")

epochs = 100
batch_size = 256
LR_H1 = 0.05
LR_H2 = 0.05

start_time = time.time()

try:
    for epoch in range(epochs):
        # Shuffle dataset each epoch for proper SGD
        indices = np.random.permutation(num_train)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, num_train, batch_size):
            X_batch = x_train_shuffled[i:i+batch_size]
            Y_batch = y_train_shuffled[i:i+batch_size]
            batch_n = len(X_batch)
            
            # 1. Forward Pass
            M, A = forward_pass(X_batch, H1, H2)
            
            # 2. Backward Pass (Blame Assignment)
            error_vector = (A - Y_batch) / batch_n  # Mean error gradient
            
            dH2 = np.dot(M.T, error_vector)
            hidden_error = np.dot(error_vector, H2.T)
            
            # Since M = ReLU(X*H1 + pool_conv_flat), the gradient wrt H1 is just 
            # X.T dot (hidden_error * relu_deriv). All 36 neurons are trainable.
            relu_deriv = (M > 0).astype(float)
            hidden_error_pre_relu = hidden_error * relu_deriv
            
            dH1 = np.dot(X_batch.T, hidden_error_pre_relu)
            
            # 3. Apply Gradients
            H1 -= LR_H1 * dH1
            H2 -= LR_H2 * dH2
            
            # Safety check (on small slice)
            if np.isnan(np.sum(H1)) or np.isnan(np.sum(H2)):
                print(f"\nGradient explosion detected. Restoring last valid model.")
                save_model(best_H1, best_H2)
                raise StopIteration
                
        # Calculate full train metrics for epoch logging
        _, train_A = forward_pass(x_train, H1, H2)
        train_loss, train_acc = get_metrics(train_A, y_train)
        
        _, test_A = forward_pass(x_test, H1, H2)
        test_loss, test_acc = get_metrics(test_A, y_test)
        
        current_time = time.time() - start_time
        '''
        # Calculate Relativistic Energy based on test accuracy
        v = (test_acc - 0.1) / 0.9
        if v < 0: v = 0
        if v >= 0.9999: v = 0.9999
        energy = (1.0 / np.sqrt(1.0 - v**2)) - 1.0
        '''
        # 1. Linearly stretch 10%-100% to 0.0-1.0
        v = (test_acc - 0.1) / 0.9
        v = np.clip(v, 0, 0.9999) # Prevent v=1.0 which causes math error

        # 2. Map 0-1 to the 0 to pi/2 range
        # We use a slightly smaller multiplier (e.g., 1.57 ≈ pi/2) 
        # to keep the energy from hitting absolute infinity too early.
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
        '''
        # Dynamic Stopping Check (Sliding Window & Escape Velocity)
        if test_acc > 0.9 and current_time > 5:
            target_time = current_time - 5.0
            idx = np.argmin(np.abs(np.array(time_history) - target_time))
            past_time = time_history[idx]
            past_energy = energy_history[idx]
            
            if past_time >= 5.0:
                expected_energy = (energy - past_energy) * (current_time / past_time)
                print(f"[Sliding Window] T: {current_time:.1f}s (vs {past_time:.1f}s) | Energy: {energy-past_energy:.4f} (Threshold: {expected_energy:.4f})")
                if energy - past_energy < expected_energy:
                    print("Stopping condition met! Energy growth is slower than time growth.")
                    save_model(best_H1, best_H2)
                    raise StopIteration
        '''
        # Epoch completed successfully
        best_H1 = copy.deepcopy(H1)
        best_H2 = copy.deepcopy(H2)
        
except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    save_model(best_H1, best_H2)
    plt.ioff()
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

print("\nTraining loop finished. Saving final model...")
save_model(best_H1, best_H2)

print("\n--- Final Test Set Evaluation ---")
_, test_A = forward_pass(x_test, best_H1, best_H2)
final_loss, final_acc = get_metrics(test_A, y_test)
print(f"Test Accuracy: {final_acc*100:.2f}%")

plt.ioff()
print("Close the plot window to exit the script.")
plt.savefig("v1.4_training_plot.png")

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
