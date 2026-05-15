import numpy as np
import os

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
init_and_save((7, 1, 3, 3), 9, "conv1.txt")
init_and_save((7, 1, 3, 3), 9, "conv2.txt")
init_and_save((7*576, 24), 576, "H1_1.txt")
init_and_save((7*288, 12), 288, "H1_2.txt")
init_and_save((7*192, 8), 192, "H1_3.txt")
init_and_save((7*96, 4), 96, "H1_4.txt")
init_and_save((7*72, 3), 72, "H1_5.txt")
init_and_save((7*48, 2), 48, "H1_6.txt")
init_and_save((7*24, 1), 24, "H1_7.txt")
init_and_save((24, 16), 24, "H2.txt")
init_and_save((784, 47), 784, "H3.txt")
np.savetxt(os.path.join(hidden_layer_dir, "residual.txt"), np.full(784, 0.5), fmt='%f')

print("Initialized model weights for v3.1 successfully.")
