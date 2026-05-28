"""
numberAI v2.1 – run_model_emnist.py
====================================
Inference script using the PyTorch-trained v2.1 EMNIST weights.
Preserves the same custom convolution / max-pooling / forward pass logic.
"""

import sys
import os
import cv2
import numpy as np
import torch
import time

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))

# ─── Load EMNIST labels ───────────────────────────────────────────────────────
labels_map = {}
labels_path = os.path.join(script_dir, "..", "emnist_labels.txt")
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                try:
                    idx = int(parts[0])
                    val = parts[1].strip()
                    labels_map[idx] = val
                except ValueError:
                    pass

def get_class_label(class_idx: int):
    return labels_map.get(class_idx, str(class_idx))


# ─── Load config & weights ───────────────────────────────────────────────────
def load_weights():
    cfg_path = os.path.join(script_dir, "hiddenLayer_emnist", "config.pt")
    K_path   = os.path.join(script_dir, "hiddenLayer_emnist", "K.pt")
    H1_path  = os.path.join(script_dir, "hiddenLayer_emnist", "H1.pt")
    H2_path  = os.path.join(script_dir, "hiddenLayer_emnist", "H2.pt")

    for p in (cfg_path, K_path, H1_path, H2_path):
        if not os.path.exists(p):
            print(f"Error: Weight file not found: {p}\n"
                  "Run reset_model_emnist.py then train_model_emnist.py first.")
            sys.exit(1)

    cfg = torch.load(cfg_path, map_location="cpu", weights_only=True)
    K   = torch.load(K_path,   map_location=DEVICE, weights_only=True)
    H1  = torch.load(H1_path,  map_location=DEVICE, weights_only=True)
    H2  = torch.load(H2_path,  map_location=DEVICE, weights_only=True)
    return cfg, K, H1, H2


# ─── Custom ops (single-image inference versions) ────────────────────────────
def conv2d_single(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    nk = K.shape[0]
    X = X.unsqueeze(0)                              # (1,28,28)
    patches = X.unfold(1, 3, 1).unfold(2, 3, 1)    # (1,26,26,3,3)
    patches = patches.contiguous().view(676, 9)     # (676,9)
    K_flat  = K.view(nk, 9)                         # (nk,9)
    C = (patches @ K_flat.T).T.view(nk, 26, 26)    # (nk,26,26)
    return C


def maxpool2d_single(C: torch.Tensor) -> torch.Tensor:
    nk = C.shape[0]
    return C.view(nk, 13, 2, 13, 2).amax(dim=(2, 4))


def run_inference(image_path: str, K, H1, H2, quiet: bool = False):
    if not os.path.exists(image_path):
        return None

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert if background is bright (ink=bright, bg=dark)
    corners = [float(image[r, c]) for r, c in [(0,0),(0,-1),(-1,0),(-1,-1)]]
    if sum(corners) / 4.0 > 127:
        image = 255 - image

    img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        C      = conv2d_single(img_tensor, K)       # (nk,26,26)
        P      = maxpool2d_single(C)                # (nk,13,13)
        P_flat = P.flatten()                        # (nk*169,)
        M      = torch.clamp(P_flat @ H1, min=0)   # (H1_n,)
        A      = M @ H2                             # (NUM_CLASSES,)

    predicted_idx = int(A.argmax().item())
    predicted_label = get_class_label(predicted_idx)
    if not quiet:
        print(f"=> Predicted: '{predicted_label}' (class index: {predicted_idx})")
    return predicted_idx, predicted_label


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[v2.1 EMNIST] Using device: {DEVICE}")
    cfg, K, H1, H2 = load_weights()
    NUM_CLASSES = cfg.get("NUM_CLASSES", 47)
    print(f"Loaded model: {cfg['NUM_KERNELS']} kernels | "
          f"H1={list(H1.shape)} | H2={list(H2.shape)}")
          
    total_params = K.nelement() + H1.nelement() + H2.nelement()
    total_size_kb = (K.element_size() * K.nelement() + H1.element_size() * H1.nelement() + H2.element_size() * H2.nelement()) / 1024.0
    print(f"Model Size: {total_params:,} parameters ({total_size_kb:.2f} KB)")

    if len(sys.argv) > 1:
        t0 = time.perf_counter()
        run_inference(sys.argv[1], K, H1, H2, quiet=False)
        t1 = time.perf_counter()
        print(f"Inference Time: {(t1 - t0) * 1000:.2f} ms")
    else:
        # Load EMNIST test set to evaluate accuracy directly
        print("\nEvaluating on full EMNIST test set...")
        data_path = os.path.join(script_dir, "..", "mnist", "emnist_28x28.npz")
        if not os.path.exists(data_path):
            print("Error: Dataset file not found.")
            sys.exit(1)
        
        data = np.load(data_path)
        x_test_np  = data['x_test'].astype(np.float32)
        y_test_np  = data['y_test'].astype(np.float32)

        x_test_full = torch.from_numpy(x_test_np).to(DEVICE)
        y_test_full = torch.from_numpy(y_test_np).to(DEVICE)

        # Batch-wise evaluation to prevent VRAM OOM
        correct = 0
        total = x_test_full.shape[0]
        batch_size = 1000
        
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, total, batch_size):
                X_b = x_test_full[i:i+batch_size]
                Y_b = y_test_full[i:i+batch_size]
                
                # unfold patches in batch
                B_b = X_b.shape[0]
                X_img = X_b.view(B_b, 28, 28)
                
                # Manual conv2d batch
                patches = X_img.unfold(1, 3, 1).unfold(2, 3, 1)          # (B, 26, 26, 3, 3)
                patches = patches.contiguous().view(B_b, 26 * 26, 9)    # (B, 676, 9)
                K_flat = K.view(cfg["NUM_KERNELS"], 9)
                C = torch.matmul(patches, K_flat.T)
                C = C.permute(0, 2, 1).contiguous().view(B_b, cfg["NUM_KERNELS"], 26, 26)
                
                # Manual maxpool
                C_r = C.view(B_b, cfg["NUM_KERNELS"], 13, 2, 13, 2)
                P = C_r.amax(dim=(3, 5))
                
                # MLP
                P_flat  = P.view(B_b, -1)
                M       = torch.clamp(P_flat @ H1, min=0)
                A       = M @ H2
                
                preds = A.argmax(dim=1)
                truths = Y_b.argmax(dim=1)
                correct += (preds == truths).sum().item()
        t1 = time.perf_counter()
        
        print(f"\n[EMNIST ByMerge] Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
        print(f"[EMNIST ByMerge] Evaluation Time: {t1 - t0:.4f}s ({(t1 - t0)*1000/total:.4f} ms/image)")
