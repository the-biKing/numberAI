import numpy as np

# Generate a 4096x128 matrix with random numbers
matrix = np.random.rand(4096, 128)

# Save to a text file
np.savetxt("matrix.txt", matrix)