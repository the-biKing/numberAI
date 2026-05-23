import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "hiddenLayer")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# He initialization
H1 = np.random.randn(784, 28) * np.sqrt(2. / 784)
H1_1 = np.random.randn(28, 4) * np.sqrt(2. / 28)
H1_2 = np.random.randn(28, 4) * np.sqrt(2. / 28)
H1_3 = np.random.randn(28, 4) * np.sqrt(2. / 28)
H1_4 = np.random.randn(28, 4) * np.sqrt(2. / 28)
H1_5 = np.random.randn(14, 2) * np.sqrt(2. / 14)
H1_6 = np.random.randn(14, 2) * np.sqrt(2. / 14)
H1_7 = np.random.randn(14, 2) * np.sqrt(2. / 14)
H1_8 = np.random.randn(14, 2) * np.sqrt(2. / 14)
H1_9 = np.random.randn(14, 2) * np.sqrt(2. / 14)
H1_10 = np.random.randn(14, 2) * np.sqrt(2. / 14)
H1_11 = np.random.randn(14, 2) * np.sqrt(2. / 14)
H2 = np.random.randn(336, 47) * np.sqrt(2. / 336)

# Save using the absolute paths
np.savetxt(os.path.join(target_dir, "H1.txt"), H1)
np.savetxt(os.path.join(target_dir, "H1_1.txt"), H1_1)
np.savetxt(os.path.join(target_dir, "H1_2.txt"), H1_2)
np.savetxt(os.path.join(target_dir, "H1_3.txt"), H1_3)
np.savetxt(os.path.join(target_dir, "H1_4.txt"), H1_4)
np.savetxt(os.path.join(target_dir, "H1_5.txt"), H1_5)
np.savetxt(os.path.join(target_dir, "H1_6.txt"), H1_6)
np.savetxt(os.path.join(target_dir, "H1_7.txt"), H1_7)
np.savetxt(os.path.join(target_dir, "H1_8.txt"), H1_8)
np.savetxt(os.path.join(target_dir, "H1_9.txt"), H1_9)
np.savetxt(os.path.join(target_dir, "H1_10.txt"), H1_10)
np.savetxt(os.path.join(target_dir, "H1_11.txt"), H1_11)
np.savetxt(os.path.join(target_dir, "H2.txt"), H2)

# Cleanup old files
for old_file in ["H_conv.txt", "H3.txt"]:
    p = os.path.join(target_dir, old_file)
    if os.path.exists(p):
        try: os.remove(p)
        except: pass

print(f"Weights reset successfully in: {target_dir}")
