"""
numberAI v2.1 – train_model.py
================================
PyTorch-accelerated training that preserves the EXACT same custom
convolution / max-pooling / hand-written backpropagation as v2.0.

What changed vs v2.0
--------------------
* All NumPy arrays replaced by torch.Tensor (GPU or CPU).
* conv2d  – still a manual 3×3 sliding window, but uses torch.Tensor
  operations (vectorised via im2col / unfolding) instead of Python loops.
* maxpool – same mask-based gradient routing, now fully vectorised in PyTorch.
* Backward pass – gradients computed by hand (no autograd).
* Multiple kernels (NUM_KERNELS) instead of a single kernel, giving the
  model the ability to learn diverse spatial features simultaneously.
* Weight checkpoints stored as .pt files for fast load/save.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("TKAgg")          # change to "Qt5Agg" if TkAgg is unavailable
import matplotlib.pyplot as plt
import copy
import os
import sys
import time

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[v2.1] Using device: {DEVICE}")

RESET = True

script_dir = os.path.dirname(os.path.abspath(__file__))

# ─── Dataset ─────────────────────────────────────────────────────────────────
print("Loading MNIST 28×28 dataset…")
data_path = os.path.join(script_dir, "..", "mnist", "mnist_28x28.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found. Run prepare_mnist.py first.")
    sys.exit(1)

data = np.load(data_path)
# Keep on CPU as numpy for the dataset; move batches to DEVICE during training
x_train_np = data['x_train'].astype(np.float32)
y_train_np = data['y_train'].astype(np.float32)
x_test_np  = data['x_test'].astype(np.float32)
y_test_np  = data['y_test'].astype(np.float32)

num_train = len(x_train_np)
print(f"Loaded {num_train} training examples and {len(x_test_np)} test examples.")

# ─── Model weights paths ─────────────────────────────────────────────────────
H1_path  = os.path.join(script_dir, "hiddenLayer", "H1.pt")
H2_path  = os.path.join(script_dir, "hiddenLayer", "H2.pt")
K_path   = os.path.join(script_dir, "hiddenLayer", "K.pt")
cfg_path = os.path.join(script_dir, "hiddenLayer", "config.pt")

if RESET:
    os.system(f"python \"{os.path.join(script_dir, 'reset_model.py')}\"")

try:
    K   = torch.load(K_path,   map_location=DEVICE, weights_only=True)
    H1  = torch.load(H1_path,  map_location=DEVICE, weights_only=True)
    H2  = torch.load(H2_path,  map_location=DEVICE, weights_only=True)
    cfg = torch.load(cfg_path, map_location="cpu",  weights_only=True)
except OSError:
    print("Error: Could not find model weights. Run reset_model.py first.")
    sys.exit(1)

NUM_KERNELS = cfg["NUM_KERNELS"]   # e.g. 8
POOL_SIZE   = 13 * 13              # 13×13 after 2×2 max-pool on 26×26
FLAT_IN     = NUM_KERNELS * POOL_SIZE

print(f"Architecture: {NUM_KERNELS} kernels | H1={H1.shape} | H2={H2.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# Custom operations  (same logic as v2.0, now in PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

def conv2d_batch(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Manual 2-D convolution (valid), multi-kernel, fully vectorised.

    X : (B, 28, 28)
    K : (NUM_KERNELS, 3, 3)
    returns C : (B, NUM_KERNELS, 26, 26)

    Implementation: im2col via unfold – exactly equivalent to sliding a 3×3
    window, but executed as a single batched matrix multiply on the GPU.
    The arithmetic is identical to the Python loop in v2.0:
        C[b, k, i, j] = sum( X[b, i:i+3, j:j+3] * K[k] )
    """
    B = X.shape[0]
    # unfold turns the 28×28 image into overlapping 3×3 patches
    # shape after unfold: (B, 9, 26*26)
    patches = X.unfold(1, 3, 1).unfold(2, 3, 1)          # (B, 26, 26, 3, 3)
    patches = patches.contiguous().view(B, 26 * 26, 9)    # (B, 676, 9)

    K_flat = K.view(NUM_KERNELS, 9)                        # (K, 9)

    # (B, 676, 9) @ (9, K) -> (B, 676, K)
    C = torch.matmul(patches, K_flat.T)                    # (B, 676, K)
    C = C.permute(0, 2, 1).contiguous().view(B, NUM_KERNELS, 26, 26)
    return C


def maxpool2d_batch(C: torch.Tensor):
    """
    2×2 max-pooling on each feature map, with gradient mask (same as v2.0).

    C    : (B, NUM_KERNELS, 26, 26)
    P    : (B, NUM_KERNELS, 13, 13)
    mask : (B, NUM_KERNELS, 26, 26)  – routes gradients back to max positions
    """
    B, nk, h, w = C.shape                                 # 26, 26
    # reshape into 2×2 blocks
    C_r = C.view(B, nk, 13, 2, 13, 2)                     # (B, K, 13, 2, 13, 2)
    P   = C_r.amax(dim=(3, 5))                             # (B, K, 13, 13)

    # build mask (handle ties by averaging, same as v2.0)
    P_broad = P[:, :, :, None, :, None]                   # (B, K, 13, 1, 13, 1)
    mask = (C_r == P_broad).float()
    mask = mask / mask.sum(dim=(3, 5), keepdim=True)
    mask = mask.view(B, nk, 26, 26)
    return P, mask


def forward_pass(X_flat: torch.Tensor, K_t: torch.Tensor,
                 H1_t: torch.Tensor, H2_t: torch.Tensor):
    """
    Full forward pass – identical logic to v2.0, extended for NUM_KERNELS.

    X_flat : (B, 784)
    returns : (X_img, mask, P_flat, M, A)
    """
    B = X_flat.shape[0]
    X_img = X_flat.view(B, 28, 28)

    C       = conv2d_batch(X_img, K_t)            # (B, K, 26, 26)
    P, mask = maxpool2d_batch(C)                  # (B, K, 13, 13)

    P_flat  = P.view(B, FLAT_IN)                  # (B, K*169)
    M       = torch.clamp(P_flat @ H1_t, min=0)  # ReLU  (B, H1)
    A       = M @ H2_t                            # logits (B, 10)
    return X_img, mask, P_flat, M, A


def get_metrics(A_vals: torch.Tensor, expected: torch.Tensor):
    """
    Euclidean loss, mean entropy, accuracy – identical formulae to v2.0.
    Works on DEVICE tensors; returns plain Python floats.
    """
    # Mean Euclidean loss
    loss = torch.mean(torch.sqrt(torch.sum((A_vals - expected) ** 2, dim=1))).item()

    # Mean entropy (softmax-based)
    logits_shifted = A_vals - A_vals.max(dim=1, keepdim=True).values
    probs   = torch.softmax(logits_shifted, dim=1)
    entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-9), dim=1)).item()

    # Accuracy
    preds  = A_vals.argmax(dim=1)
    truths = expected.argmax(dim=1)
    acc    = (preds == truths).float().mean().item()
    return loss, entropy, acc


def save_model(H1_t: torch.Tensor, H2_t: torch.Tensor, K_t: torch.Tensor):
    torch.save(H1_t.cpu(), H1_path)
    torch.save(H2_t.cpu(), H2_path)
    torch.save(K_t.cpu(),  K_path)
    print("Model saved to disk successfully.")


# ─── Pre-load full datasets to DEVICE for epoch-level evaluation ──────────────
x_train_full = torch.from_numpy(x_train_np).to(DEVICE)
y_train_full = torch.from_numpy(y_train_np).to(DEVICE)
x_test_full  = torch.from_numpy(x_test_np).to(DEVICE)
y_test_full  = torch.from_numpy(y_test_np).to(DEVICE)

# ─── Live plot setup ──────────────────────────────────────────────────────────
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("numberAI v2.1 – PyTorch Training Monitor", fontsize=14, fontweight='bold')
ax1, ax2, ax3, ax4 = axs.flatten()

loss_history       = []
entropy_history    = []
train_acc_history  = []
test_acc_history   = []

line_loss1,     = ax1.plot([], [], color='royalblue', linewidth=2, label='Train Loss')
line_loss2,     = ax2.plot([], [], color='royalblue', linewidth=2, label='Train Loss')
line_entropy,   = ax3.plot([], [], color='tomato',    linewidth=2, label='Mean Entropy')
line_train_acc, = ax4.plot([], [], color='seagreen',  linewidth=2, label='Train Acc')
line_test_acc,  = ax4.plot([], [], color='darkorange', linewidth=2, linestyle='--', label='Test Acc')

for ax, title, ylabel in [
    (ax1, "Training Loss (Normal Scale)", "Loss"),
    (ax2, "Training Loss (Log Scale)",    "Loss"),
    (ax3, "Prediction Confidence ↓",      "Entropy"),
    (ax4, "Model Accuracy",               "Accuracy"),
]:
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
ax2.set_yscale("log")
plt.tight_layout()
plt.show(block=False)

# ─── Training hyperparameters ─────────────────────────────────────────────────
epochs     = 100
batch_size = 256
# Learning rates are smaller than v2.0 because the wider network (8 kernels,
# FLAT_IN=1352) produces proportionally larger raw gradient magnitudes.
LR_H1      = 0.005
LR_H2      = 0.005
LR_K       = 0.0005
GRAD_CLIP  = 5.0       # max gradient L2-norm before clipping

best_K  = K.clone()
best_H1 = H1.clone()
best_H2 = H2.clone()

start_time = time.time()
print("Starting training (v2.1 PyTorch)… Press Ctrl+C to stop early.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════
try:
    for epoch in range(epochs):
        # Shuffle dataset
        perm          = torch.randperm(num_train)
        x_train_shuf  = x_train_full[perm]
        y_train_shuf  = y_train_full[perm]

        for i in range(0, num_train, batch_size):
            X_batch = x_train_shuf[i:i + batch_size]   # (B, 784)
            Y_batch = y_train_shuf[i:i + batch_size]   # (B, 10)
            B       = X_batch.shape[0]

            # ── 1. Forward pass ──────────────────────────────────────────────
            X_img, mask, P_flat, M, A = forward_pass(X_batch, K, H1, H2)

            # ── 2. Backward pass (hand-written, identical logic to v2.0) ─────

            # Error gradient at output layer
            error_vector = (A - Y_batch) / B             # (B, 10)

            # Output → Hidden (H2)
            dH2          = M.T @ error_vector             # (H1_n, 10)
            hidden_error = error_vector @ H2.T            # (B, H1_n)

            # Hidden → Pooled Map (H1)
            relu_deriv          = (M > 0).float()         # (B, H1_n)
            hidden_error_pre    = hidden_error * relu_deriv
            dH1                 = P_flat.T @ hidden_error_pre   # (FLAT_IN, H1_n)

            # Propagate error from H1 → Pooled Map
            delta_P_flat = hidden_error_pre @ H1.T        # (B, FLAT_IN)
            delta_P      = delta_P_flat.view(B, NUM_KERNELS, 13, 13)

            # Pooled Map → Conv Map via max-pool mask
            # Expand delta_P to match the 2×2 blocks, then apply the mask
            delta_P_r = delta_P.unsqueeze(3).unsqueeze(5)          # (B, K, 13, 1, 13, 1)
            delta_P_r = delta_P_r.expand(B, NUM_KERNELS, 13, 2, 13, 2)
            delta_C   = (delta_P_r * mask.view(B, NUM_KERNELS, 13, 2, 13, 2))
            delta_C   = delta_C.contiguous().view(B, NUM_KERNELS, 26, 26)

            # Conv Map → Kernel (dK)
            # For each kernel k:  dK[k] = sum over batch & spatial positions
            #   of delta_C[:,k,i,j] * X_img[:,i:i+3, j:j+3]
            # Vectorised via im2col:
            patches = X_img.unfold(1, 3, 1).unfold(2, 3, 1)       # (B, 26, 26, 3, 3)
            patches = patches.contiguous().view(B, 676, 9)          # (B, 676, 9)
            dc_flat = delta_C.view(B, NUM_KERNELS, 676)             # (B, K, 676)
            # dK[k, :] = sum_b sum_p  dc_flat[b,k,p] * patches[b,p,:]
            # Divide by B so the kernel gradient is a per-sample mean,
            # matching the normalisation already applied to dH1 / dH2.
            dK = torch.einsum('bkp, bpn -> kn', dc_flat, patches).view(NUM_KERNELS, 3, 3) / B

            # ── 3. Gradient clipping (L2 norm per parameter) ─────────────────
            for grad in (dH1, dH2, dK):
                gnorm = grad.norm()
                if gnorm > GRAD_CLIP:
                    grad.mul_(GRAD_CLIP / gnorm)

            # ── 4. Apply gradients ────────────────────────────────────────────
            H1 -= LR_H1 * dH1
            H2 -= LR_H2 * dH2
            K  -= LR_K  * dK

            # Safety check – detect NaN / Inf
            if (not torch.isfinite(H1).all() or
                not torch.isfinite(H2).all() or
                not torch.isfinite(K).all()):
                print("\nGradient explosion detected. Restoring last valid model.")
                save_model(best_H1, best_H2, best_K)
                raise StopIteration

        # ── Epoch-level metrics ───────────────────────────────────────────────
        with torch.no_grad():
            *_, train_A = forward_pass(x_train_full, K, H1, H2)
            *_, test_A  = forward_pass(x_test_full,  K, H1, H2)

        train_loss, train_entropy, train_acc = get_metrics(train_A, y_train_full)
        test_loss,  test_entropy,  test_acc  = get_metrics(test_A,  y_test_full)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:03d}/{epochs} "
              f"| Train Loss: {train_loss:.4f} "
              f"| Train Acc: {train_acc*100:.2f}% "
              f"| Test Acc: {test_acc*100:.2f}% "
              f"| Elapsed: {elapsed:.1f}s")

        # ── Update live plots ─────────────────────────────────────────────────
        loss_history.append(train_loss)
        entropy_history.append(train_entropy)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        xs = list(range(1, len(loss_history) + 1))

        for line in (line_loss1, line_loss2):
            line.set_xdata(xs)
            line.set_ydata(loss_history)
        line_entropy.set_xdata(xs);   line_entropy.set_ydata(entropy_history)
        line_train_acc.set_xdata(xs); line_train_acc.set_ydata(train_acc_history)
        line_test_acc.set_xdata(xs);  line_test_acc.set_ydata(test_acc_history)

        for ax in (ax1, ax2, ax3, ax4):
            ax.relim()
            ax.autoscale_view()

        plt.pause(0.01)

        # Checkpoint best weights after each clean epoch
        best_H1 = H1.clone()
        best_H2 = H2.clone()
        best_K  = K.clone()

except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current valid model…")
    save_model(best_H1, best_H2, best_K)
    plt.ioff()
    plt.show()
    sys.exit(0)

# ─── Final evaluation & save ──────────────────────────────────────────────────
total_time = time.time() - start_time
print(f"\nTraining completed in {total_time:.2f} seconds.")

save_model(best_H1, best_H2, best_K)

print("\n--- Final Test Set Evaluation ---")
with torch.no_grad():
    *_, test_A = forward_pass(x_test_full, best_K, best_H1, best_H2)
_, _, final_acc = get_metrics(test_A, y_test_full)
print(f"Test Accuracy: {final_acc * 100:.2f}%")

plt.ioff()
print("Close the plot window to exit.")
plt.show()
