import numpy as np

# Generate a 4096x128 matrix with random numbers
matrix = np.random.rand(4096, 128)

# Save to a text file
np.savetxt("H1.txt", matrix)

matrix = np.random.rand(128, 10)

# Save to a text file
np.savetxt("H2.txt", matrix)
