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

print("Resetting v3.1 PyTorch model weights...")

# Clean up any old sliced weight files to avoid confusion
for pattern in ["H1_*.txt", "H2_*.txt"]:
    for f in glob.glob(os.path.join(hidden_layer_dir, pattern)):
        try:
            os.remove(f)
        except OSError:
            pass

# conv1.txt shape: (7, 1, 3, 3) (flattened to 7x9)
init_and_save((7, 1, 3, 3), 9, "conv1.txt")

# conv2.txt shape: (7, 1, 3, 3) (flattened to 7x9)
init_and_save((7, 1, 3, 3), 9, "conv2.txt")

# H1.txt shape: (63 * 576, 12), with fan_in 576 (flattened to 36288x12)
init_and_save((63 * 576, 12), 576, "H1.txt")

# H2.txt shape: (756, 128), with fan_in 756
init_and_save((756, 128), 756, "H2.txt")

# H3.txt shape: (128, 47), with fan_in 128
init_and_save((128, 47), 128, "H3.txt")

print("Initialized model weights for v3.1 successfully.")
