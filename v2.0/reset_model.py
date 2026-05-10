import numpy as np
import os

# 1. Get the absolute path of the folder where THIS script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define the target directory relative to the script
target_dir = os.path.join(script_dir, "hiddenLayer")

# 3. Safety Check: Create the folder if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Generate matrices (Using He initialization as you planned)
H1 = np.random.randn(169, 32) * np.sqrt(2. / 169)
H2 = np.random.randn(32, 10) * np.sqrt(2. / 32)
K = np.random.randn(3, 3) * np.sqrt(2. / 9)

# 4. Save using the absolute paths
np.savetxt(os.path.join(target_dir, "H1.txt"), H1)
np.savetxt(os.path.join(target_dir, "H2.txt"), H2)
np.savetxt(os.path.join(target_dir, "K.txt"), K)

print(f"Weights reset successfully in: {target_dir}")