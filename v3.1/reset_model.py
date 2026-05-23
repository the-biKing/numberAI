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
for pattern in ["H1_*.txt", "H2_*.txt"]:
    for f in glob.glob(os.path.join(hidden_layer_dir, pattern)):
        try:
            os.remove(f)
        except OSError:
            pass

# conv1.txt shape: (7, 1, 3, 3) (flattened to 7x9)
std1 = np.sqrt(2.0 / 9)
conv1_w = np.random.randn(7, 1, 3, 3) * std1
# Set channels 3-6 to fixed Gaussian Blur
for c in range(3, 7):
    conv1_w[c, 0] = GAUSSIAN_BLUR
np.savetxt(os.path.join(hidden_layer_dir, "conv1.txt"), conv1_w.reshape(-1, 9), fmt='%f')

# conv2.txt shape: (7, 1, 3, 3) (flattened to 7x9)
std2 = np.sqrt(2.0 / 9)
conv2_w = np.random.randn(7, 1, 3, 3) * std2
# Set channels 3-6 to Sobel X, Y, Diagonal 1, and Diagonal 2
conv2_w[3, 0] = SOBEL_X
conv2_w[4, 0] = SOBEL_Y
conv2_w[5, 0] = SOBEL_DIAG1
conv2_w[6, 0] = SOBEL_DIAG2
np.savetxt(os.path.join(hidden_layer_dir, "conv2.txt"), conv2_w.reshape(-1, 9), fmt='%f')

# H1.txt shape: (63 * 576, 12), with fan_in 576 (flattened to 36288x12)
init_and_save((63 * 576, 12), 576, "H1.txt")

# H2.txt shape: (756, 128), with fan_in 756
init_and_save((756, 128), 756, "H2.txt")

# H3.txt shape: (128, 47), with fan_in 128
init_and_save((128, 47), 128, "H3.txt")

print("Initialized model weights for v3.1 successfully.")
