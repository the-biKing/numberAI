import numpy as np
import copy
import os
import sys
import time
import matplotlib.pyplot as plt

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

H1_path = os.path.join(script_dir, "hiddenLayer", "H1.txt")
H1_1_path = os.path.join(script_dir, "hiddenLayer", "H1_1.txt")
H1_2_path = os.path.join(script_dir, "hiddenLayer", "H1_2.txt")
H1_3_path = os.path.join(script_dir, "hiddenLayer", "H1_3.txt")
H1_4_path = os.path.join(script_dir, "hiddenLayer", "H1_4.txt")
H1_5_path = os.path.join(script_dir, "hiddenLayer", "H1_5.txt")
H1_6_path = os.path.join(script_dir, "hiddenLayer", "H1_6.txt")
H1_7_path = os.path.join(script_dir, "hiddenLayer", "H1_7.txt")
H1_8_path = os.path.join(script_dir, "hiddenLayer", "H1_8.txt")
H1_9_path = os.path.join(script_dir, "hiddenLayer", "H1_9.txt")
H1_10_path = os.path.join(script_dir, "hiddenLayer", "H1_10.txt")
H1_11_path = os.path.join(script_dir, "hiddenLayer", "H1_11.txt")
H2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")
H3_path = os.path.join(script_dir, "hiddenLayer", "H3.txt")

if RESET:
    os.system(f"python {os.path.join(script_dir, 'reset_model.py')}")

try:
    H1 = np.loadtxt(H1_path)
    H1_1 = np.loadtxt(H1_1_path)
    H1_2 = np.loadtxt(H1_2_path)
    H1_3 = np.loadtxt(H1_3_path)
    H1_4 = np.loadtxt(H1_4_path)
    H1_5 = np.loadtxt(H1_5_path)
    H1_6 = np.loadtxt(H1_6_path)
    H1_7 = np.loadtxt(H1_7_path)
    H1_8 = np.loadtxt(H1_8_path)
    H1_9 = np.loadtxt(H1_9_path)
    H1_10 = np.loadtxt(H1_10_path)
    H1_11 = np.loadtxt(H1_11_path)
    H2 = np.loadtxt(H2_path)
    H3 = np.loadtxt(H3_path)
except OSError:
    print("Error: Could not find model weights. Please run reset_model.py first.")
    sys.exit(1)

def forward_pass(X, H1_mat, H1_1_mat, H1_2_mat, H1_3_mat, H1_4_mat, H1_5_mat, H1_6_mat, 
                 H1_7_mat, H1_8_mat, H1_9_mat, H1_10_mat, H1_11_mat, H2_mat, H3_mat):
    N = X.shape[0]
    
    O_H1 = np.dot(X, H1_mat)
    
    X_2d = X.reshape(N, 28, 28)
    
    X1_1 = X_2d[:, 0:7, :]
    O1_1 = np.dot(X1_1, H1_1_mat)
    
    X1_2 = X_2d[:, 7:14, :]
    O1_2 = np.dot(X1_2, H1_2_mat)
    
    X1_3 = X_2d[:, 14:21, :]
    O1_3 = np.dot(X1_3, H1_3_mat)
    
    X1_4 = X_2d[:, 21:28, :]
    O1_4 = np.dot(X1_4, H1_4_mat)
    
    X1_5 = X_2d[:, 0:14, 7:21]
    O1_5 = np.dot(X1_5, H1_5_mat)
    
    X1_6 = X_2d[:, 14:28, 7:21]
    O1_6 = np.dot(X1_6, H1_6_mat)
    
    X1_7 = X_2d[:, 0:14, 0:14] # Top-left
    O1_7 = np.dot(X1_7, H1_7_mat)
    
    X1_8 = X_2d[:, 0:14, 14:28] # Top-right
    O1_8 = np.dot(X1_8, H1_8_mat)
    
    X1_9 = X_2d[:, 14:28, 0:14] # Bottom-left
    O1_9 = np.dot(X1_9, H1_9_mat)
    
    X1_10 = X_2d[:, 14:28, 14:28] # Bottom-right
    O1_10 = np.dot(X1_10, H1_10_mat)
    
    X1_11 = X_2d[:, 7:21, 7:21] # Center
    O1_11 = np.dot(X1_11, H1_11_mat)
    
    O_H1_flat = O_H1.reshape(N, 28)
    O1_1_flat = O1_1.reshape(N, 28)
    O1_2_flat = O1_2.reshape(N, 28)
    O1_3_flat = O1_3.reshape(N, 28)
    O1_4_flat = O1_4.reshape(N, 28)
    O1_5_flat = O1_5.reshape(N, 28)
    O1_6_flat = O1_6.reshape(N, 28)
    O1_7_flat = O1_7.reshape(N, 28)
    O1_8_flat = O1_8.reshape(N, 28)
    O1_9_flat = O1_9.reshape(N, 28)
    O1_10_flat = O1_10.reshape(N, 28)
    O1_11_flat = O1_11.reshape(N, 28)
    
    concat_O = np.concatenate((O_H1_flat, O1_1_flat, O1_2_flat, O1_3_flat, O1_4_flat, O1_5_flat, O1_6_flat, 
                               O1_7_flat, O1_8_flat, O1_9_flat, O1_10_flat, O1_11_flat), axis=1)
    M1 = np.maximum(0, concat_O)
    
    O_H2 = np.dot(M1, H2_mat)
    M2 = np.maximum(0, O_H2)
    A = np.dot(M2, H3_mat)
    
    return O_H1, O1_1, O1_2, O1_3, O1_4, O1_5, O1_6, O1_7, O1_8, O1_9, O1_10, O1_11, M1, O_H2, M2, A

def get_metrics(A_vals, expected):
    loss = np.mean(np.sqrt(np.sum((A_vals - expected)**2, axis=1)))
    preds = np.argmax(A_vals, axis=1)
    truths = np.argmax(expected, axis=1)
    acc = np.mean(preds == truths)
    return loss, acc

def save_model(H1_mat, H1_1_mat, H1_2_mat, H1_3_mat, H1_4_mat, H1_5_mat, H1_6_mat, 
               H1_7_mat, H1_8_mat, H1_9_mat, H1_10_mat, H1_11_mat, H2_mat, H3_mat):
    np.savetxt(H1_path, H1_mat, fmt='%f')
    np.savetxt(H1_1_path, H1_1_mat, fmt='%f')
    np.savetxt(H1_2_path, H1_2_mat, fmt='%f')
    np.savetxt(H1_3_path, H1_3_mat, fmt='%f')
    np.savetxt(H1_4_path, H1_4_mat, fmt='%f')
    np.savetxt(H1_5_path, H1_5_mat, fmt='%f')
    np.savetxt(H1_6_path, H1_6_mat, fmt='%f')
    np.savetxt(H1_7_path, H1_7_mat, fmt='%f')
    np.savetxt(H1_8_path, H1_8_mat, fmt='%f')
    np.savetxt(H1_9_path, H1_9_mat, fmt='%f')
    np.savetxt(H1_10_path, H1_10_mat, fmt='%f')
    np.savetxt(H1_11_path, H1_11_mat, fmt='%f')
    np.savetxt(H2_path, H2_mat, fmt='%f')
    np.savetxt(H3_path, H3_mat, fmt='%f')
    print("Model saved to disk successfully.")

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

best_H1 = copy.deepcopy(H1)
best_H1_1 = copy.deepcopy(H1_1)
best_H1_2 = copy.deepcopy(H1_2)
best_H1_3 = copy.deepcopy(H1_3)
best_H1_4 = copy.deepcopy(H1_4)
best_H1_5 = copy.deepcopy(H1_5)
best_H1_6 = copy.deepcopy(H1_6)
best_H1_7 = copy.deepcopy(H1_7)
best_H1_8 = copy.deepcopy(H1_8)
best_H1_9 = copy.deepcopy(H1_9)
best_H1_10 = copy.deepcopy(H1_10)
best_H1_11 = copy.deepcopy(H1_11)
best_H2 = copy.deepcopy(H2)
best_H3 = copy.deepcopy(H3)

print("Starting training with Mini-batch Backpropagation... Press Ctrl+C to stop early.")

epochs = 100
batch_size = 256
LR_H1 = 0.05
LR_H2 = 0.05
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
            
            O_H1, O1_1, O1_2, O1_3, O1_4, O1_5, O1_6, O1_7, O1_8, O1_9, O1_10, O1_11, M1, O_H2, M2, A = forward_pass(
                X_batch, H1, H1_1, H1_2, H1_3, H1_4, H1_5, H1_6, H1_7, H1_8, H1_9, H1_10, H1_11, H2, H3)
            
            error_vector = (A - Y_batch) / batch_n
            
            dH3 = np.dot(M2.T, error_vector)
            
            hidden_error_m2 = np.dot(error_vector, H3.T)
            relu_deriv_m2 = (M2 > 0).astype(float)
            hidden_error_pre_relu_m2 = hidden_error_m2 * relu_deriv_m2
            
            dH2 = np.dot(M1.T, hidden_error_pre_relu_m2)
            
            hidden_error_m1 = np.dot(hidden_error_pre_relu_m2, H2.T)
            relu_deriv_m1 = (M1 > 0).astype(float)
            hidden_error_pre_relu = hidden_error_m1 * relu_deriv_m1
            
            dO_H1_flat = hidden_error_pre_relu[:, 0:28]
            dO1_1_flat = hidden_error_pre_relu[:, 28:56]
            dO1_2_flat = hidden_error_pre_relu[:, 56:84]
            dO1_3_flat = hidden_error_pre_relu[:, 84:112]
            dO1_4_flat = hidden_error_pre_relu[:, 112:140]
            dO1_5_flat = hidden_error_pre_relu[:, 140:168]
            dO1_6_flat = hidden_error_pre_relu[:, 168:196]
            dO1_7_flat = hidden_error_pre_relu[:, 196:224]
            dO1_8_flat = hidden_error_pre_relu[:, 224:252]
            dO1_9_flat = hidden_error_pre_relu[:, 252:280]
            dO1_10_flat = hidden_error_pre_relu[:, 280:308]
            dO1_11_flat = hidden_error_pre_relu[:, 308:336]
            
            dO_H1 = dO_H1_flat.reshape(O_H1.shape)
            dO1_1 = dO1_1_flat.reshape(O1_1.shape)
            dO1_2 = dO1_2_flat.reshape(O1_2.shape)
            dO1_3 = dO1_3_flat.reshape(O1_3.shape)
            dO1_4 = dO1_4_flat.reshape(O1_4.shape)
            dO1_5 = dO1_5_flat.reshape(O1_5.shape)
            dO1_6 = dO1_6_flat.reshape(O1_6.shape)
            dO1_7 = dO1_7_flat.reshape(O1_7.shape)
            dO1_8 = dO1_8_flat.reshape(O1_8.shape)
            dO1_9 = dO1_9_flat.reshape(O1_9.shape)
            dO1_10 = dO1_10_flat.reshape(O1_10.shape)
            dO1_11 = dO1_11_flat.reshape(O1_11.shape)
            
            dH1 = np.dot(X_batch.T, dO_H1)
            
            X_2d = X_batch.reshape(batch_n, 28, 28)
            dH1_1 = np.tensordot(X_2d[:, 0:7, :], dO1_1, axes=([0, 1], [0, 1]))
            dH1_2 = np.tensordot(X_2d[:, 7:14, :], dO1_2, axes=([0, 1], [0, 1]))
            dH1_3 = np.tensordot(X_2d[:, 14:21, :], dO1_3, axes=([0, 1], [0, 1]))
            dH1_4 = np.tensordot(X_2d[:, 21:28, :], dO1_4, axes=([0, 1], [0, 1]))
            dH1_5 = np.tensordot(X_2d[:, 0:14, 7:21], dO1_5, axes=([0, 1], [0, 1]))
            dH1_6 = np.tensordot(X_2d[:, 14:28, 7:21], dO1_6, axes=([0, 1], [0, 1]))
            dH1_7 = np.tensordot(X_2d[:, 0:14, 0:14], dO1_7, axes=([0, 1], [0, 1]))
            dH1_8 = np.tensordot(X_2d[:, 0:14, 14:28], dO1_8, axes=([0, 1], [0, 1]))
            dH1_9 = np.tensordot(X_2d[:, 14:28, 0:14], dO1_9, axes=([0, 1], [0, 1]))
            dH1_10 = np.tensordot(X_2d[:, 14:28, 14:28], dO1_10, axes=([0, 1], [0, 1]))
            dH1_11 = np.tensordot(X_2d[:, 7:21, 7:21], dO1_11, axes=([0, 1], [0, 1]))
            
            H1 -= LR_H1 * dH1
            H1_1 -= LR_H1 * dH1_1
            H1_2 -= LR_H1 * dH1_2
            H1_3 -= LR_H1 * dH1_3
            H1_4 -= LR_H1 * dH1_4
            H1_5 -= LR_H1 * dH1_5
            H1_6 -= LR_H1 * dH1_6
            H1_7 -= LR_H1 * dH1_7
            H1_8 -= LR_H1 * dH1_8
            H1_9 -= LR_H1 * dH1_9
            H1_10 -= LR_H1 * dH1_10
            H1_11 -= LR_H1 * dH1_11
            H2 -= LR_H2 * dH2
            H3 -= LR_H3 * dH3
            
            if np.isnan(np.sum(H3)):
                print(f"\nGradient explosion detected. Restoring last valid model.")
                save_model(best_H1, best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H1_5, best_H1_6, 
                           best_H1_7, best_H1_8, best_H1_9, best_H1_10, best_H1_11, best_H2, best_H3)
                raise StopIteration
                
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, train_A = forward_pass(x_train, H1, H1_1, H1_2, H1_3, H1_4, H1_5, H1_6, 
                                                                      H1_7, H1_8, H1_9, H1_10, H1_11, H2, H3)
        train_loss, train_acc = get_metrics(train_A, y_train)
        
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, test_A = forward_pass(x_test, H1, H1_1, H1_2, H1_3, H1_4, H1_5, H1_6, 
                                                                     H1_7, H1_8, H1_9, H1_10, H1_11, H2, H3)
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
        
        best_H1 = copy.deepcopy(H1)
        best_H1_1 = copy.deepcopy(H1_1)
        best_H1_2 = copy.deepcopy(H1_2)
        best_H1_3 = copy.deepcopy(H1_3)
        best_H1_4 = copy.deepcopy(H1_4)
        best_H1_5 = copy.deepcopy(H1_5)
        best_H1_6 = copy.deepcopy(H1_6)
        best_H1_7 = copy.deepcopy(H1_7)
        best_H1_8 = copy.deepcopy(H1_8)
        best_H1_9 = copy.deepcopy(H1_9)
        best_H1_10 = copy.deepcopy(H1_10)
        best_H1_11 = copy.deepcopy(H1_11)
        best_H2 = copy.deepcopy(H2)
        best_H3 = copy.deepcopy(H3)
        
except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    save_model(best_H1, best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H1_5, best_H1_6, 
               best_H1_7, best_H1_8, best_H1_9, best_H1_10, best_H1_11, best_H2, best_H3)
    plt.ioff()
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

print("\nTraining loop finished. Saving final model...")
save_model(best_H1, best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H1_5, best_H1_6, 
           best_H1_7, best_H1_8, best_H1_9, best_H1_10, best_H1_11, best_H2, best_H3)

print("\n--- Final Test Set Evaluation ---")
_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, test_A = forward_pass(x_test, best_H1, best_H1_1, best_H1_2, best_H1_3, best_H1_4, best_H1_5, best_H1_6, 
                                                             best_H1_7, best_H1_8, best_H1_9, best_H1_10, best_H1_11, best_H2, best_H3)
final_loss, final_acc = get_metrics(test_A, y_test)
print(f"Test Accuracy: {final_acc*100:.2f}%")

plt.ioff()
print("Close the plot window to exit the script.")
plt.savefig("v1.8.2_training_plot.png")

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
