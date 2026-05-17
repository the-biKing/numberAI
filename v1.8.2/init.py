import numpy as np

# Generate a 784x256 matrix with random numbers
matrix = np.random.rand(784, 256)
np.savetxt("./pyScript/hiddenLayer/H1.txt", matrix)

matrix = np.random.rand(256, 64)
np.savetxt("./pyScript/hiddenLayer/H2.txt", matrix)

matrix = np.random.rand(64, 10)
np.savetxt("./pyScript/hiddenLayer/H3.txt", matrix)
