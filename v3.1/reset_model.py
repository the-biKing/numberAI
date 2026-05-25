import numpy as np
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
hidden_layer_dir = os.path.join(script_dir, "hiddenLayer")

if not os.path.exists(hidden_layer_dir):
    os.makedirs(hidden_layer_dir)

# Helper function to initialize and save weights
def init_and_save(shape, fan_in, filename):
    # He Initialization
    std = np.sqrt(2.0 / fan_in)
    W = np.random.randn(*shape) * std
    np.savetxt(os.path.join(hidden_layer_dir, filename), W.reshape(-1, W.shape[-1]), fmt='%f')

# Fixed 3x3 Gaussian Blur kernel for Step 1 (conv1)
GAUSSIAN_BLUR = np.array([
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
], dtype=np.float32)

# Fixed 3x3 kernels for Step 2 (conv2)
SOBEL_X = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.float32)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

SOBEL_DIAG1 = np.array([
    [-2, -1,  0],
    [-1,  0,  1],
    [ 0,  1,  2]
], dtype=np.float32)

SOBEL_DIAG2 = np.array([
    [ 0, -1, -2],
    [ 1,  0, -1],
    [ 2,  1,  0]
], dtype=np.float32)

print("Resetting v3.1 PyTorch model weights...")

# Clean up any old sliced weight files to avoid confusion
for pattern in ["H1_*.txt", "H2_*.txt", "H1.txt", "H2.txt", "H3.txt", "conv1.txt", "conv2.txt"]:
    for f in glob.glob(os.path.join(hidden_layer_dir, pattern)):
        try:
            os.remove(f)
        except OSError:
            pass

# conv1.txt shape: (4, 1, 3, 3) (flattened to 4x9)
std1 = np.sqrt(2.0 / 9)
conv1_w = np.random.randn(4, 1, 3, 3) * std1
np.savetxt(os.path.join(hidden_layer_dir, "conv1.txt"), conv1_w.reshape(-1, 9), fmt='%f')

# conv2.txt shape: (4, 1, 3, 3) (flattened to 4x9)
std2 = np.sqrt(2.0 / 9)
conv2_w = np.random.randn(4, 1, 3, 3) * std2
np.savetxt(os.path.join(hidden_layer_dir, "conv2.txt"), conv2_w.reshape(-1, 9), fmt='%f')

# H1_trainable.txt shape: (36 * 576, 12), with fan_in 576 (flattened to 20736x12)
init_and_save((36 * 576, 12), 576, "H1_trainable.txt")

# H1_fixed.txt shape: (4 * 26, 12), with fan_in 26 (flattened to 104x12)
init_and_save((4 * 26, 12), 26, "H1_fixed.txt")

# H2.txt shape: (480, 128), with fan_in 480
init_and_save((480, 128), 480, "H2.txt")

# H3.txt shape: (128, 47), with fan_in 128
init_and_save((128, 47), 128, "H3.txt")

print("Initialized model weights for v3.1 successfully.")
