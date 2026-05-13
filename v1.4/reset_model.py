import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "hiddenLayer")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# He initialization
H1 = np.random.randn(256, 36) * np.sqrt(2. / 256)
H2 = np.random.randn(36, 10) * np.sqrt(2. / 36)

# Save using the absolute paths
np.savetxt(os.path.join(target_dir, "H1.txt"), H1)
np.savetxt(os.path.join(target_dir, "H2.txt"), H2)

print(f"Weights reset successfully in: {target_dir}")
