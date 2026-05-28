"""
numberAI v2.1 – train_model_emnist.py
======================================
PyTorch-accelerated training that preserves the EXACT same custom
convolution / max-pooling / hand-written backpropagation as v2.1,
but adapted for EMNIST ByMerge (47 classes).
Uses separate hiddenLayer_emnist weights folder and safe batch evaluation.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend to prevent display errors
import matplotlib.pyplot as plt
import copy
import os
import sys
import time
import csv

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[v2.1 EMNIST] Using device: {DEVICE}")

RESET = True

script_dir = os.path.dirname(os.path.abspath(__file__))

# ─── Dataset ─────────────────────────────────────────────────────────────────
print("Loading EMNIST ByMerge 28×28 dataset…")
data_path = os.path.join(script_dir, "..", "mnist", "emnist_28x28.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found. Run prepare_emnist.py first.")
    sys.exit(1)

data = np.load(data_path)
x_train_np = data['x_train'].astype(np.float32)
y_train_np = data['y_train'].astype(np.float32)
x_test_np  = data['x_test'].astype(np.float32)
y_test_np  = data['y_test'].astype(np.float32)

num_train = len(x_train_np)
print(f"Loaded {num_train} training examples and {len(x_test_np)} test examples.")

# ─── Model weights paths ─────────────────────────────────────────────────────
H1_path  = os.path.join(script_dir, "hiddenLayer_emnist", "H1.pt")
H2_path  = os.path.join(script_dir, "hiddenLayer_emnist", "H2.pt")
K_path   = os.path.join(script_dir, "hiddenLayer_emnist", "K.pt")
cfg_path = os.path.join(script_dir, "hiddenLayer_emnist", "config.pt")

if RESET:
    os.system(f"python \"{os.path.join(script_dir, 'reset_model_emnist.py')}\"")

try:
    K   = torch.load(K_path,   map_location=DEVICE, weights_only=True)
    H1  = torch.load(H1_path,  map_location=DEVICE, weights_only=True)
    H2  = torch.load(H2_path,  map_location=DEVICE, weights_only=True)
    cfg = torch.load(cfg_path, map_location="cpu",  weights_only=True)
except OSError:
    print("Error: Could not find model weights. Run reset_model_emnist.py first.")
    sys.exit(1)

NUM_KERNELS = cfg["NUM_KERNELS"]   # e.g. 8
POOL_SIZE   = 13 * 13              # 13×13 after 2×2 max-pool on 26×26
FLAT_IN     = NUM_KERNELS * POOL_SIZE
NUM_CLASSES = cfg.get("NUM_CLASSES", 47)

print(f"Architecture: {NUM_KERNELS} kernels | H1={H1.shape} | H2={H2.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# Custom operations (identical logic to v2.1)
# ═══════════════════════════════════════════════════════════════════════════════

def conv2d_batch(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    B = X.shape[0]
    patches = X.unfold(1, 3, 1).unfold(2, 3, 1)          # (B, 26, 26, 3, 3)
    patches = patches.contiguous().view(B, 26 * 26, 9)    # (B, 676, 9)
    K_flat = K.view(NUM_KERNELS, 9)                        # (K, 9)
    C = torch.matmul(patches, K_flat.T)                    # (B, 676, K)
    C = C.permute(0, 2, 1).contiguous().view(B, NUM_KERNELS, 26, 26)
    return C


def maxpool2d_batch(C: torch.Tensor):
    B, nk, h, w = C.shape
    C_r = C.view(B, nk, 13, 2, 13, 2)                     # (B, K, 13, 2, 13, 2)
    P   = C_r.amax(dim=(3, 5))                             # (B, K, 13, 13)
    P_broad = P[:, :, :, None, :, None]                   # (B, K, 13, 1, 13, 1)
    mask = (C_r == P_broad).float()
    mask = mask / mask.sum(dim=(3, 5), keepdim=True)
    mask = mask.view(B, nk, 26, 26)
    return P, mask


def forward_pass(X_flat: torch.Tensor, K_t: torch.Tensor,
                 H1_t: torch.Tensor, H2_t: torch.Tensor):
    B = X_flat.shape[0]
    X_img = X_flat.view(B, 28, 28)
    C       = conv2d_batch(X_img, K_t)            # (B, K, 26, 26)
    P, mask = maxpool2d_batch(C)                  # (B, K, 13, 13)
    P_flat  = P.view(B, FLAT_IN)                  # (B, K*169)
    M       = torch.clamp(P_flat @ H1_t, min=0)  # ReLU  (B, H1)
    A       = M @ H2_t                            # logits (B, NUM_CLASSES)
    return X_img, mask, P_flat, M, A


# ═══════════════════════════════════════════════════════════════════════════════
# Memory-Safe Batch Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_dataset(X_flat: torch.Tensor, Y_expected: torch.Tensor, K_t, H1_t, H2_t, batch_size=1000):
    total = X_flat.shape[0]
    total_loss = 0.0
    total_entropy = 0.0
    correct = 0
    with torch.no_grad():
        for i in range(0, total, batch_size):
            X_b = X_flat[i:i+batch_size]
            Y_b = Y_expected[i:i+batch_size]
            
            *_, A_b = forward_pass(X_b, K_t, H1_t, H2_t)
            
            # Euclidean loss sum for this batch
            loss_b = torch.sqrt(torch.sum((A_b - Y_b) ** 2, dim=1))
            total_loss += loss_b.sum().item()
            
            # Entropy sum
            logits_shifted = A_b - A_b.max(dim=1, keepdim=True).values
            probs = torch.softmax(logits_shifted, dim=1)
            entropy_b = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            total_entropy += entropy_b.sum().item()
            
            # Correct predictions
            preds = A_b.argmax(dim=1)
            truths = Y_b.argmax(dim=1)
            correct += (preds == truths).sum().item()
            
    return total_loss / total, total_entropy / total, correct / total


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

# ─── Live plot setup (headless) ───────────────────────────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("numberAI v2.1 EMNIST – Headless Training Monitor", fontsize=14, fontweight='bold')
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

# ─── Training hyperparameters ─────────────────────────────────────────────────
epochs     = 10  # Standard epoch count for EMNIST comparative tests
batch_size = 256
LR_H1      = 0.005
LR_H2      = 0.005
LR_K       = 0.0005
GRAD_CLIP  = 5.0       # max gradient L2-norm before clipping

best_K  = K.clone()
best_H1 = H1.clone()
best_H2 = H2.clone()

start_time = time.time()
print("Starting training (v2.1 PyTorch EMNIST ByMerge)… Press Ctrl+C to stop early.\n")


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
            Y_batch = y_train_shuf[i:i + batch_size]   # (B, 47)
            B       = X_batch.shape[0]

            # ── 1. Forward pass ──────────────────────────────────────────────
            X_img, mask, P_flat, M, A = forward_pass(X_batch, K, H1, H2)

            # ── 2. Backward pass (hand-written custom v2.1 engine) ───────────
            error_vector = (A - Y_batch) / B             # (B, 47)

            # Output → Hidden (H2)
            dH2          = M.T @ error_vector             # (H1_n, 47)
            hidden_error = error_vector @ H2.T            # (B, H1_n)

            # Hidden → Pooled Map (H1)
            relu_deriv          = (M > 0).float()         # (B, H1_n)
            hidden_error_pre    = hidden_error * relu_deriv
            dH1                 = P_flat.T @ hidden_error_pre   # (FLAT_IN, H1_n)

            # Propagate error from H1 → Pooled Map
            delta_P_flat = hidden_error_pre @ H1.T        # (B, FLAT_IN)
            delta_P      = delta_P_flat.view(B, NUM_KERNELS, 13, 13)

            # Pooled Map → Conv Map via max-pool mask
            delta_P_r = delta_P.unsqueeze(3).unsqueeze(5)
            delta_P_r = delta_P_r.expand(B, NUM_KERNELS, 13, 2, 13, 2)
            delta_C   = (delta_P_r * mask.view(B, NUM_KERNELS, 13, 2, 13, 2))
            delta_C   = delta_C.contiguous().view(B, NUM_KERNELS, 26, 26)

            # Conv Map → Kernel (dK)
            patches = X_img.unfold(1, 3, 1).unfold(2, 3, 1)       # (B, 26, 26, 3, 3)
            patches = patches.contiguous().view(B, 676, 9)          # (B, 676, 9)
            dc_flat = delta_C.view(B, NUM_KERNELS, 676)             # (B, K, 676)
            dK = torch.einsum('bkp, bpn -> kn', dc_flat, patches).view(NUM_KERNELS, 3, 3) / B

            # ── 3. Gradient clipping ─────────────────────────────────────────
            for grad in (dH1, dH2, dK):
                gnorm = grad.norm()
                if gnorm > GRAD_CLIP:
                    grad.mul_(GRAD_CLIP / gnorm)

            # ── 4. Apply gradients ────────────────────────────────────────────
            H1 -= LR_H1 * dH1
            H2 -= LR_H2 * dH2
            K  -= LR_K  * dK

            # Safety check
            if (not torch.isfinite(H1).all() or
                not torch.isfinite(H2).all() or
                not torch.isfinite(K).all()):
                print("\nGradient explosion detected. Restoring last valid model.")
                save_model(best_H1, best_H2, best_K)
                raise StopIteration

        # ── Epoch-level metrics (Memory-Safe Batch Evaluation) ────────────────
        train_loss, train_entropy, train_acc = evaluate_dataset(x_train_full, y_train_full, K, H1, H2)
        test_loss,  test_entropy,  test_acc  = evaluate_dataset(x_test_full,  y_test_full,  K, H1, H2)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:03d}/{epochs} "
              f"| Train Loss: {train_loss:.4f} "
              f"| Train Acc: {train_acc*100:.2f}% "
              f"| Test Acc: {test_acc*100:.2f}% "
              f"| Elapsed: {elapsed:.1f}s")

        # ── Update live history & save plots ──────────────────────────────────
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

        # Checkpoint best weights
        best_H1 = H1.clone()
        best_H2 = H2.clone()
        best_K  = K.clone()

except StopIteration:
    pass
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current valid model…")
    save_model(best_H1, best_H2, best_K)
    plot_save_path = os.path.join(script_dir, "..", "v2.1_emnist_training_plot.png")
    plt.savefig(plot_save_path)
    print(f"Training graph saved to: {plot_save_path}")
    sys.exit(0)

# ─── Final evaluation & save ──────────────────────────────────────────────────
total_time = time.time() - start_time
print(f"\nTraining completed in {total_time:.2f} seconds.")

save_model(best_H1, best_H2, best_K)

print("\n--- Final Test Set Evaluation ---")
final_loss, final_entropy, final_acc = evaluate_dataset(x_test_full, y_test_full, best_K, best_H1, best_H2)
print(f"Test Accuracy: {final_acc * 100:.2f}%")

plot_save_path = os.path.join(script_dir, "..", "v2.1_emnist_training_plot.png")
plt.savefig(plot_save_path)
print(f"Training graph saved to: {plot_save_path}")

# ─── Log to CSV ───────────────────────────────────────────────────────────────
csv_path = os.path.join(script_dir, "..", "benchmark_results.csv")
file_exists = os.path.isfile(csv_path)
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow(["v2.1_emnist", round(total_time, 2), round(final_loss, 4), round(final_acc*100, 2)])
print(f"Results saved to {csv_path}")
