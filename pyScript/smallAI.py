import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Generate a 4096x128 matrix with random numbers
H1 = torch.randn(256, 32, device=device, dtype=torch.float32)

matrix = np.random.rand(32, 10)

# Save to a text file
np.savetxt("./pyScript/hiddenLayer/H2.txt", matrix)

def firstmulti():   #function for first multiplication
    H1 = np.loadtxt("./pyScript/hiddenLayer/H1.txt")

    for i in range(10):

        I = np.loadtxt(f"./pyScript/inputText/{i}.txt")
        I = I.flatten()
        # Read the contents of the files
        M = np.dot(H1.T, I)
        np.savetxt(f"./pyScript/hiddenLayer/M{i}.txt", M)

def secondmulti():  #function for second multiplication
    H2 = np.loadtxt("./pyScript/hiddenLayer/H2.txt")
    for i in range(10):
        M = np.loadtxt(f"./pyScript/hiddenLayer/M{i}.txt")

        # Read the contents of the files
        A = np.dot(H2.T, M)
        np.savetxt(f"./pyScript/hiddenLayer/A{i}.txt", A)


def loss():
    Loss = 0
    for i in range(10):
    # Load the matrices from the text files
        A = np.loadtxt(f'./pyScript/hiddenLayer/A{i}.txt') 
        EA = np.loadtxt(f'./pyScript/expectedAnswer/EA{i}.txt')  

        # Calculate the Loss
        Loss += np.sqrt(np.sum((A - EA)**2))
        return Loss


firstmulti()
secondmulti()
minLoss=loss()


H1 = np.loadtxt("./pyScript/hiddenLayer/H1.txt")

LR = 0.1

for k in range(500):
    for i in range(256):
        for j in range(32):
            # Store the current loss before updating H1[i][j]
            current_loss = loss()

            # Adjust H1[i][j] based on a small step size
            H1[i,j] += 0.01
            firstmulti()
            secondmulti()

            # Calculate new loss after the change
            new_loss = loss()
            H1[i,j] -= 0.01
            H1[i,j] -= ((new_loss-current_loss)/0.01)*LR

            # Save the matrix after every update (optional, depends on your need)
            np.savetxt("./pyScript/hiddenLayer/H1.txt", H1, fmt='%f')

        # Print the current loss for tracking progress
        print(f"Iteration {k}, Loss: {new_loss}")

        


        
    
