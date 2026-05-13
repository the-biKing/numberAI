import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "hiddenLayer")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# He initialization
H1 = np.random.randn(784, 256) * np.sqrt(2. / 784)
H2 = np.random.randn(256, 64) * np.sqrt(2. / 256)
H3 = np.random.randn(64, 10) * np.sqrt(2. / 64)

# Save using the absolute paths
np.savetxt(os.path.join(target_dir, "H1.txt"), H1)
np.savetxt(os.path.join(target_dir, "H2.txt"), H2)
np.savetxt(os.path.join(target_dir, "H3.txt"), H3)

print(f"Weights reset successfully in: {target_dir}")
