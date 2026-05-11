"""
numberAI v2.1 – run_model.py
==============================
Inference script using the PyTorch-trained v2.1 weights.
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

# ─── Load config & weights ───────────────────────────────────────────────────
def load_weights():
    cfg_path = os.path.join(script_dir, "hiddenLayer", "config.pt")
    K_path   = os.path.join(script_dir, "hiddenLayer", "K.pt")
    H1_path  = os.path.join(script_dir, "hiddenLayer", "H1.pt")
    H2_path  = os.path.join(script_dir, "hiddenLayer", "H2.pt")

    for p in (cfg_path, K_path, H1_path, H2_path):
        if not os.path.exists(p):
            print(f"Error: Weight file not found: {p}\n"
                  "Run reset_model.py then train_model.py first.")
            sys.exit(1)

    cfg = torch.load(cfg_path, map_location="cpu", weights_only=True)
    K   = torch.load(K_path,   map_location=DEVICE, weights_only=True)
    H1  = torch.load(H1_path,  map_location=DEVICE, weights_only=True)
    H2  = torch.load(H2_path,  map_location=DEVICE, weights_only=True)
    return cfg, K, H1, H2


# ─── Custom ops (single-image inference versions) ────────────────────────────
def conv2d_single(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """X: (28,28)  K: (nk,3,3)  →  C: (nk,26,26)"""
    nk = K.shape[0]
    X = X.unsqueeze(0)                              # (1,28,28)
    patches = X.unfold(1, 3, 1).unfold(2, 3, 1)    # (1,26,26,3,3)
    patches = patches.contiguous().view(676, 9)     # (676,9)
    K_flat  = K.view(nk, 9)                         # (nk,9)
    C = (patches @ K_flat.T).T.view(nk, 26, 26)    # (nk,26,26)
    return C


def maxpool2d_single(C: torch.Tensor) -> torch.Tensor:
    """C: (nk,26,26)  →  P: (nk,13,13)"""
    nk = C.shape[0]
    return C.view(nk, 13, 2, 13, 2).amax(dim=(2, 4))


def run_inference(image_path: str, K, H1, H2, quiet: bool = False):
    if not os.path.exists(image_path):
        return None

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert if background is bright (MNIST convention: ink=bright, bg=dark)
    corners = [float(image[r, c]) for r, c in [(0,0),(0,-1),(-1,0),(-1,-1)]]
    if sum(corners) / 4.0 > 127:
        image = 255 - image

    img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).to(DEVICE)

    # Forward pass (no gradients needed)
    with torch.no_grad():
        C      = conv2d_single(img_tensor, K)       # (nk,26,26)
        P      = maxpool2d_single(C)                # (nk,13,13)
        P_flat = P.flatten()                        # (nk*169,)
        M      = torch.clamp(P_flat @ H1, min=0)   # (H1_n,)
        A      = M @ H2                             # (10,)

    predicted_digit = int(A.argmax().item())
    if not quiet:
        print(f"=> Predicted: {predicted_digit}")
    return predicted_digit


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[v2.1] Using device: {DEVICE}")
    cfg, K, H1, H2 = load_weights()
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
        verify_dir = os.path.join(script_dir, "..", "verify")
        train_dir  = os.path.join(script_dir, "..", "inputImage")

        def evaluate_directory(directory: str, dataset_name: str):
            if not os.path.exists(directory):
                print(f"\nError: Directory {directory} not found.")
                return
            print(f"\nEvaluating {dataset_name} dataset in "
                  f"{os.path.basename(directory)}/…")
            correct = total = 0
            
            t0 = time.perf_counter()
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path   = os.path.join(directory, filename)
                    predicted  = run_inference(img_path, K, H1, H2, quiet=True)
                    if predicted is not None:
                        true_str = filename.split('_')[0].split('.')[0]
                        if true_str.isdigit():
                            total += 1
                            if predicted == int(true_str):
                                correct += 1
            t1 = time.perf_counter()
            if total > 0:
                print(f"[{dataset_name}] Accuracy: {correct}/{total} "
                      f"({correct/total*100:.2f}%)")
                print(f"[{dataset_name}] Total Time: {t1 - t0:.4f}s "
                      f"({(t1 - t0)*1000/total:.2f} ms/image)")
            else:
                print(f"[{dataset_name}] No valid images found.")

        evaluate_directory(train_dir, "Training")
        evaluate_directory(verify_dir, "Verify")
