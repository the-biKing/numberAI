import torch
import os

# ─────────────────────────────────────────────────────────────────────────────
# v2.1 reset_model_emnist.py
# Initialises weights for the PyTorch-accelerated model adapted for EMNIST ByMerge.
# Architecture:
#   Input  : 28x28 image
#   Conv   : NUM_KERNELS × 3x3 learnable kernels  →  NUM_KERNELS × 26x26
#   MaxPool: 2x2                                  →  NUM_KERNELS × 13x13
#   Flatten: NUM_KERNELS × 169
#   H1     : (NUM_KERNELS*169, 64)  – ReLU
#   H2     : (64, 47)               – raw logits (47 classes for EMNIST ByMerge)
# ─────────────────────────────────────────────────────────────────────────────

NUM_KERNELS = 8          # number of convolutional kernels (v2.0 used 1)
H1_NEURONS  = 64        # hidden layer width
NUM_CLASSES = 47        # 47 classes for EMNIST ByMerge

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "hiddenLayer_emnist")
os.makedirs(target_dir, exist_ok=True)

flat_in = NUM_KERNELS * 169          # 13*13 = 169 per kernel

# He (Kaiming) initialisation – same strategy as v2.0
K  = torch.randn(NUM_KERNELS, 3, 3) * (2.0 / 9)  ** 0.5
H1 = torch.randn(flat_in, H1_NEURONS) * (2.0 / flat_in) ** 0.5
H2 = torch.randn(H1_NEURONS, NUM_CLASSES) * (2.0 / H1_NEURONS) ** 0.5

torch.save(K,  os.path.join(target_dir, "K.pt"))
torch.save(H1, os.path.join(target_dir, "H1.pt"))
torch.save(H2, os.path.join(target_dir, "H2.pt"))

# Also save config so other scripts know the shape
config = {"NUM_KERNELS": NUM_KERNELS, "H1_NEURONS": H1_NEURONS, "NUM_CLASSES": NUM_CLASSES}
torch.save(config, os.path.join(target_dir, "config.pt"))

print(f"Weights reset for EMNIST ByMerge  →  K:{list(K.shape)}  H1:{list(H1.shape)}  H2:{list(H2.shape)}")
print(f"Saved in: {target_dir}")
