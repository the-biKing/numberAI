import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import time

RESET = True

# 1. Preload everything to avoid disk I/O bottleneck
inputs = []
expected_answers = []

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "inputText")

print("Loading training data...")
for filename in os.listdir(input_dir):
    if not filename.endswith('.txt'):
        continue
        
    I_path = os.path.join(input_dir, filename)
    
    # Extract the digit label from the filename (e.g., '4_7.txt' -> 4)
    digit_str = filename.split('_')[0].split('.')[0]
    digit = int(digit_str)
    
    EA_path = os.path.join(script_dir, "..", "expectedAnswer", f"EA{digit}.txt")
    
    # Load input images (1D array of 256 size)
    I = np.loadtxt(I_path).flatten()
    inputs.append(I)
    
    # Load expected answers (1D array of 10 size)
    EA = np.loadtxt(EA_path)
    expected_answers.append(EA)

inputs = np.array(inputs)/255.0 #normalize inputs
expected_answers = np.array(expected_answers) # Shape: (N, 10)
num_examples = len(inputs)
print(f"Loaded {num_examples} training examples.")

if RESET:
    os.system(f"python {script_dir}\\reset_model.py")

H1_path = os.path.join(script_dir, "hiddenLayer", "H1.txt")
H2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")

try:
    H1 = np.loadtxt(H1_path)
    H2 = np.loadtxt(H2_path)
except OSError:
    print("Error: Could not find H1.txt or H2.txt. Please run reset_model.py first.")
    sys.exit(1)

# In-memory intermediate values
M_vals = np.zeros((num_examples, 32))
A_vals = np.zeros((num_examples, 10))

def forward_pass(H1_mat, H2_mat):
    global M_vals, A_vals
    M_vals = np.maximum(0, np.dot(inputs, H1_mat))
    A_vals = np.dot(M_vals, H2_mat)

def get_loss():
    # Mean Squared Error (MSE)
    return np.mean((A_vals - expected_answers)**2)

def get_entropy():
    exp_A = np.exp(A_vals - np.max(A_vals, axis=1, keepdims=True))
    probs = exp_A / np.sum(exp_A, axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    return np.mean(entropy)

def save_model(H1_mat, H2_mat):
    np.savetxt(H1_path, H1_mat, fmt='%f')
    np.savetxt(H2_path, H2_mat, fmt='%f')
    print("Model saved to disk successfully.")

# Initialize dynamic plot
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Subplot 1: Loss
loss_history = []
line_loss, = ax1.plot(loss_history, label='Total Loss', color='b', linewidth=2)
ax1.set_xlabel("Iteration (k)")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.grid(True)
ax1.legend()

# Subplot 2: Entropy
entropy_history = []
line_entropy, = ax2.plot(entropy_history, label='Mean Entropy', color='r', linewidth=2)
ax2.set_xlabel("Iteration (k)")
ax2.set_ylabel("Entropy")
ax2.set_title("Prediction Confidence (Lower = More Confident)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show(block=False)

best_H1 = copy.deepcopy(H1)
best_H2 = copy.deepcopy(H2)

print("Starting training with Backpropagation... Press Ctrl+C to stop early and save the model.")
try:
    # Increased iterations since backprop is extremely fast. 
    # You can stop early via Ctrl+C if the plot levels out!
    start_time = time.time()
    for k in range(100):
        # We might need larger learning rates for true backprop, but sticking to your base ones.
        LR_H1 = 0.01
        LR_H2 = 0.001
        
        # 1. Forward Pass
        forward_pass(H1, H2)
        iter_loss = get_loss()
        iter_entropy = get_entropy()
        
        if np.isnan(iter_loss) or np.isinf(iter_loss):
            print(f"\nGradient explosion detected (NaN/Inf).")
            print("Restoring last valid model and stopping.")
            save_model(best_H1, best_H2)
            raise StopIteration
            
        # 2. Backward Pass (Blame Assignment / Backpropagation)
        # Calculate the error vector (A - Expected)
        # We divide by num_examples to average the gradient and keep updates stable
        error_vector = (A_vals - expected_answers) / num_examples 
        
        # Update H2: transpose M * error vector
        dH2 = np.dot(M_vals.T, error_vector)  # Shape: (32, 10)
        
        # Backpropagate error to hidden layer: error vector * transpose H2
        hidden_error = np.dot(error_vector, H2.T)  # Shape: (N, 32)
        
        # Apply ReLU derivative (1 if M > 0 else 0)
        relu_deriv = (M_vals > 0).astype(float)
        hidden_error_pre_relu = hidden_error * relu_deriv
        
        # Update H1: transpose inputs * hidden error (with relu applied)
        dH1 = np.dot(inputs.T, hidden_error_pre_relu)  # Shape: (256, 32)
        
        # 3. Apply the Gradients
        H1 -= LR_H1 * dH1
        H2 -= LR_H2 * dH2

        print(f"Iteration {k}, Loss: {iter_loss:.6f}, Entropy: {iter_entropy:.6f}")
        
        # Update plots
        loss_history.append(iter_loss)
        line_loss.set_ydata(loss_history)
        line_loss.set_xdata(range(len(loss_history)))
        ax1.relim()
        ax1.autoscale_view()
        
        entropy_history.append(iter_entropy)
        line_entropy.set_ydata(entropy_history)
        line_entropy.set_xdata(range(len(entropy_history)))
        ax2.relim()
        ax2.autoscale_view()
        
        # Pause to let the plot update dynamically
        plt.pause(0.01)
        
        # Iteration completed successfully
        best_H1 = copy.deepcopy(H1)
        best_H2 = copy.deepcopy(H2)
    
    end_time = time.time()
    print(f"Training finished in {end_time - start_time} seconds.")
    
except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    save_model(best_H1, best_H2)
    plt.ioff()
    sys.exit(0)

print("\nTraining loop finished. Saving final model...")
save_model(best_H1, best_H2)

plt.ioff()
print("Close the plot window to exit the script.")
plt.savefig("v1.1_training_plot.png")

import csv
import time
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
ver = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
t_time = time.time() - start_time
f_loss = iter_loss if 'iter_loss' in locals() else "N/A"
f_acc = "N/A"
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow([ver, round(t_time, 2), round(f_loss, 4) if isinstance(f_loss, float) else f_loss, f_acc])
print(f"Results saved to {csv_path}")
