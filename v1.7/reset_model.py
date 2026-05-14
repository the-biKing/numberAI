import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "hiddenLayer")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# He initialization
H1_1 = np.random.randn(784, 28) * np.sqrt(2. / 784)
H1_2 = np.random.randn(392, 14) * np.sqrt(2. / 392)
H1_3 = np.random.randn(112, 4) * np.sqrt(2. / 112)
H1_4 = np.random.randn(28, 1) * np.sqrt(2. / 28)
H2 = np.random.randn(112, 49) * np.sqrt(2. / 112)
H_conv = np.random.randn(16, 1) * np.sqrt(2. / 16)
H3 = np.random.randn(49, 10) * np.sqrt(2. / 49)

# Save using the absolute paths
np.savetxt(os.path.join(target_dir, "H1_1.txt"), H1_1)
np.savetxt(os.path.join(target_dir, "H1_2.txt"), H1_2)
np.savetxt(os.path.join(target_dir, "H1_3.txt"), H1_3)
np.savetxt(os.path.join(target_dir, "H1_4.txt"), H1_4)
np.savetxt(os.path.join(target_dir, "H2.txt"), H2)
np.savetxt(os.path.join(target_dir, "H_conv.txt"), H_conv)
np.savetxt(os.path.join(target_dir, "H3.txt"), H3)

# Cleanup old files
for old_file in ["H1.txt"]:
    p = os.path.join(target_dir, old_file)
    if os.path.exists(p):
        try: os.remove(p)
        except: pass

print(f"Weights reset successfully in: {target_dir}")
