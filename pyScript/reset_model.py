import numpy as np

# Generate a 256x32 and 32x10 matrix with random numbers
H1 = np.random.rand(256, 32)
H2 = np.random.rand(32, 10)

# Save to a text file
np.savetxt("./pyScript/hiddenLayer/H1.txt", H1)
np.savetxt("./pyScript/hiddenLayer/H2.txt", H2)

print("Neural network weights (H1, H2) have been reset with random values.")
