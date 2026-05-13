import numpy as np

# Generate a 784x64 matrix with random numbers
matrix = np.random.rand(784, 64)

# Save to a text file
np.savetxt("./pyScript/hiddenLayer/H1.txt", matrix)

matrix = np.random.rand(64, 10)

# Save to a text file
np.savetxt("./pyScript/hiddenLayer/H2.txt", matrix)
