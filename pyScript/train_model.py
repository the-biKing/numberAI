import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys

# 1. Preload everything to avoid disk I/O bottleneck
inputs = []
expected_answers = []

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "inputText")

print("Loading training data...")
for filename in os.listdir(input_dir):
    if not filename.endswith('.txt'):
        continue
        
    I_path = os.path.join(input_dir, filename)
    
    # Extract the digit label from the filename (e.g., '4_7.txt' -> 4, '3.txt' -> 3)
    digit_str = filename.split('_')[0].split('.')[0]
    digit = int(digit_str)
    
    EA_path = os.path.join(script_dir, "expectedAnswer", f"EA{digit}.txt")
    
    # Load input images (1D array of 256 size)
    I = np.loadtxt(I_path).flatten()
    inputs.append(I)
    
    # Load expected answers (1D array of 10 size)
    EA = np.loadtxt(EA_path)
    expected_answers.append(EA)

inputs = np.array(inputs) # Shape: (N, 256)
expected_answers = np.array(expected_answers) # Shape: (N, 10)
num_examples = len(inputs)
print(f"Loaded {num_examples} training examples.")

H1_path = os.path.join(script_dir, "hiddenLayer", "H1.txt")
H2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")

try:
    H1 = np.loadtxt(H1_path)
    H2 = np.loadtxt(H2_path)
except OSError:
    print("Error: Could not find H1.txt or H2.txt. Please run reset_model.py first.")
    sys.exit(1)

# In-memory intermediate values instead of files
M_vals = np.zeros((num_examples, 32))
A_vals = np.zeros((num_examples, 10))

def forward_pass(H1_mat, H2_mat):
    global M_vals, A_vals
    # Vectorized forward pass for all examples at once
    M_vals = np.maximum(0, np.dot(inputs, H1_mat))
    A_vals = np.dot(M_vals, H2_mat)

def get_loss():
    # Sum of Euclidean distances for each example
    return np.sum(np.sqrt(np.sum((A_vals - expected_answers)**2, axis=1)))

def save_model(H1_mat, H2_mat):
    np.savetxt(H1_path, H1_mat, fmt='%f')
    np.savetxt(H2_path, H2_mat, fmt='%f')
    print("Model saved to disk successfully.")

# Initialize dynamic plot
plt.ion()
fig, ax = plt.subplots()
loss_history = []
line, = ax.plot(loss_history, label='Total Loss', color='b', linewidth=2)
ax.set_xlabel("Iteration (k)")
ax.set_ylabel("Loss")
ax.set_title("Training Loss (Dynamic)")
ax.grid(True)
ax.legend()
plt.show(block=False)

best_H1 = copy.deepcopy(H1)
best_H2 = copy.deepcopy(H2)

print("Starting training... Press Ctrl+C to stop early and save the model.")
try:
    for k in range(500):
        LR_H1 = 0.02
        LR_H2 = 0.002
        
        # Optimize H1
        for i in range(256):
            for j in range(32):
                forward_pass(H1, H2)
                current_loss = get_loss()

                H1[i][j] += 0.01
                forward_pass(H1, H2)
                new_loss = get_loss()
                
                if np.isnan(new_loss) or np.isinf(new_loss):
                    print(f"\nGradient explosion detected (NaN/Inf) at H1[{i}][{j}].")
                    print("Restoring last valid model and stopping.")
                    save_model(best_H1, best_H2)
                    raise StopIteration
                
                H1[i][j] -= 0.01
                adjust = ((new_loss - current_loss) / 0.01) * LR_H1
                H1[i][j] -= adjust

        # Optimize H2
        for i in range(32):
            for j in range(10):
                forward_pass(H1, H2)
                current_loss = get_loss()

                H2[i][j] += 0.01
                forward_pass(H1, H2)
                new_loss = get_loss()
                
                if np.isnan(new_loss) or np.isinf(new_loss):
                    print(f"\nGradient explosion detected (NaN/Inf) at H2[{i}][{j}].")
                    print("Restoring last valid model and stopping.")
                    save_model(best_H1, best_H2)
                    raise StopIteration

                H2[i][j] -= 0.01
                adjust = ((new_loss - current_loss) / 0.01) * LR_H2
                H2[i][j] -= adjust

        # Calculate final loss for this iteration
        forward_pass(H1, H2)
        iter_loss = get_loss()
        print(f"Iteration {k}, Loss: {iter_loss:.6f}")
        
        # Update plot
        loss_history.append(iter_loss)
        line.set_ydata(loss_history)
        line.set_xdata(range(len(loss_history)))
        ax.relim()
        ax.autoscale_view()
        # Pause to let the plot update
        plt.pause(0.01)
        
        # Iteration completed successfully without exploding gradients
        best_H1 = copy.deepcopy(H1)
        best_H2 = copy.deepcopy(H2)
        
except StopIteration:
    # Triggered by gradient explosion
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    save_model(best_H1, best_H2)
    plt.ioff()
    plt.show()
    sys.exit(0)

print("\nTraining loop finished. Saving final model...")
save_model(best_H1, best_H2)

plt.ioff()
print("Close the plot window to exit the script.")
plt.show()
