import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import time

RESET = True

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading MNIST 28x28 dataset...")
data_path = os.path.join(script_dir, "..", "mnist", "mnist_28x28.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found. Run prepare_mnist.py first.")
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
H2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")

if RESET:
    os.system(f"python {os.path.join(script_dir, 'reset_model.py')}")

try:
    H1_1 = np.loadtxt(H1_1_path)
    H1_2 = np.loadtxt(H1_2_path)
    H1_3 = np.loadtxt(H1_3_path)
    H1_4 = np.loadtxt(H1_4_path)
    H2 = np.loadtxt(H2_path)
except OSError:
    print("Error: Could not find model weights. Please run reset_model.py first.")
    sys.exit(1)

def forward_pass(X, H1_1_mat, H1_2_mat, H1_3_mat, H1_4_mat, H2_mat):
    N = X.shape[0]
    
    O1 = np.dot(X, H1_1_mat)
    
    X2 = X.reshape(N, 2, 392)
    O2 = np.dot(X2, H1_2_mat)
    
    X3 = X.reshape(N, 7, 112)
    O3 = np.dot(X3, H1_3_mat)
    
    X4 = X.reshape(N, 28, 28)
    O4 = np.dot(X4, H1_4_mat)
    
    O1_flat = O1.reshape(N, 28)
    O2_flat = O2.reshape(N, 28)
    O3_flat = O3.reshape(N, 28)
    O4_flat = O4.reshape(N, 28)
    
    concat_O = np.concatenate((O1_flat, O2_flat, O3_flat, O4_flat), axis=1)
    M1 = np.maximum(0, concat_O)
    
    A = np.dot(M1, H2_mat)
    return O1, O2, O3, O4, M1, A

def get_metrics(A_vals, expected):
    # Mean Euclidean loss
    loss = np.mean(np.sqrt(np.sum((A_vals - expected)**2, axis=1)))
    
    # Accuracy
    preds = np.argmax(A_vals, axis=1)
    truths = np.argmax(expected, axis=1)
    acc = np.mean(preds == truths)
    return loss, acc

def save_model(H1_1_mat, H1_2_mat, H1_3_mat, H1_4_mat, H2_mat):
    np.savetxt(H1_1_path, H1_1_mat, fmt='%f')
    np.savetxt(H1_2_path, H1_2_mat, fmt='%f')
    np.savetxt(H1_3_path, H1_3_mat, fmt='%f')
    np.savetxt(H1_4_path, H1_4_mat, fmt='%f')
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

best_H1_1 = copy.deepcopy(H1_1)
best_H1_2 = copy.deepcopy(H1_2)
best_H1_3 = copy.deepcopy(H1_3)
best_H1_4 = copy.deepcopy(H1_4)
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
            O1, O2, O3, O4, M1, A = forward_pass(X_batch, H1_1, H1_2, H1_3, H1_4, H2)
            
            # 2. Backward Pass (Blame Assignment)
            error_vector = (A - Y_batch) / batch_n  # Mean error gradient
            
            dH2 = np.dot(M1.T, error_vector)
            
            hidden_error_m1 = np.dot(error_vector, H2.T)
            relu_deriv = (M1 > 0).astype(float)
            hidden_error_pre_relu = hidden_error_m1 * relu_deriv
            
            dO1_flat = hidden_error_pre_relu[:, 0:28]
            dO2_flat = hidden_error_pre_relu[:, 28:56]
            dO3_flat = hidden_error_pre_relu[:, 56:84]
            dO4_flat = hidden_error_pre_relu[:, 84:112]
            
            dO1 = dO1_flat.reshape(O1.shape)
            dO2 = dO2_flat.reshape(O2.shape)
            dO3 = dO3_flat.reshape(O3.shape)
            dO4 = dO4_flat.reshape(O4.shape)
            
            dH1_1 = np.dot(X_batch.T, dO1)
            
            X2 = X_batch.reshape(batch_n, 2, 392)
            dH1_2 = np.tensordot(X2, dO2, axes=([0, 1], [0, 1]))
            
            X3 = X_batch.reshape(batch_n, 7, 112)
            dH1_3 = np.tensordot(X3, dO3, axes=([0, 1], [0, 1]))
            
            X4 = X_batch.reshape(batch_n, 28, 28)
            dH1_4 = np.tensordot(X4, dO4, axes=([0, 1], [0, 1]))
            
            # 3. Apply Gradients
            H1_1 -= LR_H1 * dH1_1
            H1_2 -= LR_H1 * dH1_2
            H1_3 -= LR_H1 * dH1_3
            H1_4 -= LR_H1 * dH1_4
            H2 -= LR_H2 * dH2
            
            # Safety check (on small slice)
            if np.isnan(np.sum(H2)):
                print(f"\nGradient explosion detected. Restoring last valid model.")
                save_model(best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H2)
                raise StopIteration
                
        # Calculate full train metrics for epoch logging
        _, _, _, _, _, train_A = forward_pass(x_train, H1_1, H1_2, H1_3, H1_4, H2)
        train_loss, train_acc = get_metrics(train_A, y_train)
        
        _, _, _, _, _, test_A = forward_pass(x_test, H1_1, H1_2, H1_3, H1_4, H2)
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
        best_H1_1 = copy.deepcopy(H1_1)
        best_H1_2 = copy.deepcopy(H1_2)
        best_H1_3 = copy.deepcopy(H1_3)
        best_H1_4 = copy.deepcopy(H1_4)
        best_H2 = copy.deepcopy(H2)
        
except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    save_model(best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H2)
    plt.ioff()
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

print("\nTraining loop finished. Saving final model...")
save_model(best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H2)

print("\n--- Final Test Set Evaluation ---")
_, _, _, _, _, test_A = forward_pass(x_test, best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H2)
final_loss, final_acc = get_metrics(test_A, y_test)
print(f"Test Accuracy: {final_acc*100:.2f}%")

plt.ioff()
print("Close the plot window to exit the script.")
plt.savefig("v1.2_training_plot.png")

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
