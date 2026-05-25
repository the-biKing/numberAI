import numpy as np
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
hidden_layer_dir = os.path.join(script_dir, "hiddenLayer")

if not os.path.exists(hidden_layer_dir):
    os.makedirs(hidden_layer_dir)

# Helper function to initialize and save weights
def init_and_save(shape, fan_in, filename):
    std = np.sqrt(2.0 / fan_in)
    W = np.random.randn(*shape) * std
    np.savetxt(os.path.join(hidden_layer_dir, filename), W.reshape(-1, W.shape[-1]), fmt='%f')

print("Resetting distilled student weights...")

# Clean up old weight files
for f in glob.glob(os.path.join(hidden_layer_dir, "*.txt")):
    try:
        os.remove(f)
    except OSError:
        pass

# 1. conv1.txt shape: (4, 1, 3, 3) -> flattened to 4x9
std1 = np.sqrt(2.0 / 9)
conv1_w = np.random.randn(4, 1, 3, 3) * std1
np.savetxt(os.path.join(hidden_layer_dir, "conv1.txt"), conv1_w.reshape(-1, 9), fmt='%f')

# 2. conv2.txt shape: (4, 1, 3, 3) -> flattened to 4x9
std2 = np.sqrt(2.0 / 9)
conv2_w = np.random.randn(4, 1, 3, 3) * std2
np.savetxt(os.path.join(hidden_layer_dir, "conv2.txt"), conv2_w.reshape(-1, 9), fmt='%f')

# 3. H1_trainable.txt shape: (36 * 576, 12) -> flattened to 20736x12
init_and_save((36 * 576, 12), 576, "H1_trainable.txt")

# 4. H1_fixed.txt shape: (4 * 26, 12) -> flattened to 104x12
init_and_save((4 * 26, 12), 26, "H1_fixed.txt")

# 5. H2.txt shape: (480, 128)
init_and_save((480, 128), 480, "H2.txt")

# 6. H3.txt shape: (128, 47)
init_and_save((128, 47), 128, "H3.txt")

print("Distilled student weights reset successfully.")
