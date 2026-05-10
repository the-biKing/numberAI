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
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

num_train = len(x_train)
print(f"Loaded {num_train} training examples and {len(x_test)} test examples.")


H1_path = os.path.join(script_dir, "hiddenLayer", "H1.txt")
H2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")
K_path = os.path.join(script_dir, "hiddenLayer", "K.txt")

if RESET:
    os.system(f"python {script_dir}\\reset_model.py")
try:
    H1 = np.loadtxt(H1_path)
    H2 = np.loadtxt(H2_path)
    K = np.loadtxt(K_path)
except OSError:
    print("Error: Could not find model weights. Please run reset_model.py first.")
    sys.exit(1)

def conv2d_batch(X, K):
    batch_size = X.shape[0]
    out = np.zeros((batch_size, 26, 26))
    for i in range(26):
        for j in range(26):
            out[:, i, j] = np.sum(X[:, i:i+3, j:j+3] * K, axis=(1, 2))
    return out

def maxpool2d_batch(C):
    reshaped = C.reshape(C.shape[0], 13, 2, 13, 2)
    P = reshaped.max(axis=(2, 4))
    P_broad = P[:, :, np.newaxis, :, np.newaxis]
    mask = (reshaped == P_broad).astype(float)
    # Average duplicate maxes to conserve gradient sum
    mask /= mask.sum(axis=(2, 4), keepdims=True)
    mask = mask.reshape(C.shape[0], 26, 26)
    return P, mask

def forward_pass(X_flat, K_mat, H1_mat, H2_mat):
    batch_n = X_flat.shape[0]
    X_img = X_flat.reshape(batch_n, 28, 28)
    
    C = conv2d_batch(X_img, K_mat)
    P, mask = maxpool2d_batch(C)
    
    P_flat = P.reshape(batch_n, 169)
    M = np.maximum(0, np.dot(P_flat, H1_mat))
    A = np.dot(M, H2_mat)
    return X_img, mask, P_flat, M, A

def get_metrics(A_vals, expected):
    # Mean Euclidean loss
    loss = np.mean(np.sqrt(np.sum((A_vals - expected)**2, axis=1)))
    # loss = np.mean((A_vals - expected)**2)
    
    # Mean Entropy
    exp_A = np.exp(A_vals - np.max(A_vals, axis=1, keepdims=True))
    probs = exp_A / np.sum(exp_A, axis=1, keepdims=True)
    entropy = -np.mean(np.sum(probs * np.log(probs + 1e-9), axis=1))
    
    # Accuracy
    preds = np.argmax(A_vals, axis=1)
    truths = np.argmax(expected, axis=1)
    acc = np.mean(preds == truths)
    return loss, entropy, acc

def save_model(H1_mat, H2_mat, K_mat):
    np.savetxt(H1_path, H1_mat, fmt='%f')
    np.savetxt(H2_path, H2_mat, fmt='%f')
    np.savetxt(K_path, K_mat, fmt='%f')
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

# Subplot 3: Entropy
entropy_history = []
line_entropy, = ax3.plot(entropy_history, label='Mean Entropy', color='r', linewidth=2)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Entropy")
ax3.set_title("Prediction Confidence (Lower = More Confident)")
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
best_K = copy.deepcopy(K)

print("Starting training with Mini-batch Backpropagation... Press Ctrl+C to stop early.")

epochs = 100
batch_size = 256
# Bumped up learning rates slightly to converge nicely on batch size of 256
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
            X_img, mask, P_flat, M, A = forward_pass(X_batch, K, H1, H2)
            
            # 2. Backward Pass (Blame Assignment)
            error_vector = (A - Y_batch) / batch_n  # Mean error gradient
            
            # Output -> Hidden
            dH2 = np.dot(M.T, error_vector)
            hidden_error = np.dot(error_vector, H2.T)
            
            # Hidden -> Pooled Map
            relu_deriv = (M > 0).astype(float)
            hidden_error_pre_relu = hidden_error * relu_deriv
            dH1 = np.dot(P_flat.T, hidden_error_pre_relu)
            
            # Propagate error from Hidden to Pooled Map (13x13)
            delta_P_flat = np.dot(hidden_error_pre_relu, H1.T)
            delta_P = delta_P_flat.reshape(batch_n, 13, 13)
            
            # Pooled Map -> Conv Map (26x26) via Max-Pool mask
            delta_P_broad = delta_P[:, :, np.newaxis, :, np.newaxis]
            delta_C = (delta_P_broad * mask.reshape(batch_n, 13, 2, 13, 2)).reshape(batch_n, 26, 26)
            
            # Conv Map -> Kernel (3x3)
            dK = np.zeros((3, 3))
            for h in range(3):
                for w in range(3):
                    dK[h, w] = np.sum(delta_C * X_img[:, h:h+26, w:w+26])
            
            # 3. Apply Gradients
            H1 -= LR_H1 * dH1
            H2 -= LR_H2 * dH2
            # Add a learning rate for Kernel, let's use the same as H1
            K -= LR_H1 * dK
            
            # Safety check (on small slice)
            if np.isnan(np.sum(H1)) or np.isnan(np.sum(H2)) or np.isnan(np.sum(K)):
                print(f"\nGradient explosion detected. Restoring last valid model.")
                save_model(best_H1, best_H2, best_K)
                raise StopIteration
                
        # Calculate full train metrics for epoch logging
        *_, train_A = forward_pass(x_train, K, H1, H2)
        train_loss, train_entropy, train_acc = get_metrics(train_A, y_train)
        
        *_, test_A = forward_pass(x_test, K, H1, H2)
        test_loss, test_entropy, test_acc = get_metrics(test_A, y_test)
        
        print(f"Epoch {epoch+1:03d}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        
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
        
        entropy_history.append(train_entropy)
        line_entropy.set_ydata(entropy_history)
        line_entropy.set_xdata(range(len(entropy_history)))
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
        
        # Epoch completed successfully
        best_H1 = copy.deepcopy(H1)
        best_H2 = copy.deepcopy(H2)
        best_K = copy.deepcopy(K)
        
except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current valid model...")
    save_model(best_H1, best_H2, best_K)
    plt.ioff()
    plt.show()
    sys.exit(0)

end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")

print("\nTraining loop finished. Saving final model...")
save_model(best_H1, best_H2, best_K)

print("\n--- Final Test Set Evaluation ---")
*_, test_A = forward_pass(x_test, best_K, best_H1, best_H2)
final_loss, final_entropy, final_acc = get_metrics(test_A, y_test)
print(f"Test Accuracy: {final_acc*100:.2f}%")

plt.ioff()
print("Close the plot window to exit the script.")
plt.show()
