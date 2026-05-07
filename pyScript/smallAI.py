import numpy as np
import matplotlib.pyplot as plt



# Generate a 4096x128 matrix with random numbers
H1 = np.random.rand(256, 32)
H2 = np.random.rand(32, 10)

# Save to a text file
np.savetxt("C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/H1.txt", H1)
np.savetxt("C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/H2.txt", H2)


def firstmulti(H1):   #function for first multiplication
   

    for i in range(10):

        I = np.loadtxt(f"C:/Users/jesse/Documents/GitHub/numberAI/pyScript/inputText/{i}.txt")
        I = I.flatten()
        # Read the contents of the files
        M = np.dot(H1.T, I)
        np.savetxt(f"C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/M{i}.txt", M)

def secondmulti(H2):  #function for second multiplication

    for i in range(10):
        M = np.loadtxt(f"C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/M{i}.txt")

        # Read the contents of the files
        A = np.dot(H2.T, M)
        np.savetxt(f"C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/A{i}.txt", A)


def loss():
    Loss = 0
    for i in range(10):
    # Load the matrices from the text files
        A = np.loadtxt(f'C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/A{i}.txt') 
        EA = np.loadtxt(f'C:/Users/jesse/Documents/GitHub/numberAI/pyScript/expectedAnswer/EA{i}.txt')  

        # Calculate the Loss
        Loss += np.sqrt(np.sum((A - EA)**2))
        return Loss

def poloss(anum):
    poLoss = 0
    A = np.loadtxt(f'C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/A{anum}.txt')
    EA = np.loadtxt(f'C:/Users/jesse/Documents/GitHub/numberAI/pyScript/expectedAnswer/EA{anum}.txt')
    poLoss = np.sqrt(np.sum((A - EA)**2))
    return poLoss
 


H1=np.loadtxt("C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/H1.txt")
H2=np.loadtxt("C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/H2.txt")

firstmulti(H1)
secondmulti(H2)





for k in range(500):
    LR = 0.02
    H1=np.loadtxt("C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/H1.txt")
    for i in range(256):
        for j in range(32):
            # Store the current loss before updating H1[i][j]
            current_loss = loss()

            # Adjust H1[i][j] based on a small step size
            
            H1[i][j] += 0.01
            firstmulti(H1)
            secondmulti(H2)

            # Calculate new loss after the change
            new_loss = loss()
            H1[i][j] -= 0.01
            adjust = ((new_loss-current_loss)/0.01)*LR
            H1[i][j] -= adjust
        print(f"Iteration {k}, Loss: {new_loss}")
        # Print the current loss for tracking progress
    np.savetxt("C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/H1.txt", H1, fmt='%f')
    H2=np.loadtxt("C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/H2.txt")
    LR = 0.002
    for i in range(32):
        for j in range(10):

            current_poloss = poloss(j)

            H2[i][j] += 0.01
            firstmulti(H1)
            secondmulti(H2)

            new_poloss = poloss(j)
            H2[i][j] -= 0.01
            adjust = ((new_poloss-current_poloss)/0.01)*LR
            H2[i][j] -= adjust
        print(f"Iteration {k}, Loss: {new_loss}")
    np.savetxt("C:/Users/jesse/Documents/GitHub/numberAI/pyScript/hiddenLayer/H2.txt", H2, fmt='%f')