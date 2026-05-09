import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Generate a 256x32 and 32x10 matrix with random numbers
H1 = np.random.rand(256, 32)
H2 = np.random.rand(32, 10)

# Save to a text file
np.savetxt(os.path.join(script_dir, "hiddenLayer", "H1.txt"), H1)
np.savetxt(os.path.join(script_dir, "hiddenLayer", "H2.txt"), H2)

print("Neural network weights (H1, H2) have been reset with random values.")
