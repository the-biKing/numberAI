import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from train_model import forward_pass

N = 2
X = np.random.randn(N, 784)
Y = np.random.randn(N, 47)

H1_1 = np.random.randn(784, 56)
H1_2 = np.random.randn(392, 28)
H1_3 = np.random.randn(112, 8)
H1_4 = np.random.randn(56, 4)
H1_5 = np.random.randn(28, 2)
H2 = np.random.randn(280, 196)
H_conv = np.random.randn(3, 3)
H3 = np.random.randn(196, 47)

O1, O2, O3, O4, O5, M1, X_blocks, M2, M2_relu, A = forward_pass(X, H1_1, H1_2, H1_3, H1_4, H1_5, H2, H_conv, H3)

error_vector = (A - Y) / N
dH3 = np.dot(M2_relu.T, error_vector)
hidden_error_m2 = np.dot(error_vector, H3.T)
relu_deriv_m2 = (M2 > 0).astype(float)
hidden_error_pre_relu_m2 = hidden_error_m2 * relu_deriv_m2

dH2 = np.dot(M1.T, hidden_error_pre_relu_m2)

hidden_error_m2_spatial = hidden_error_pre_relu_m2.reshape(N, 14, 14)
dH_conv = np.tensordot(hidden_error_m2_spatial, X_blocks, axes=([0, 1, 2], [0, 1, 2]))

hidden_error_m1 = np.dot(hidden_error_pre_relu_m2, H2.T)
relu_deriv_m1 = (M1 > 0).astype(float)
hidden_error_pre_relu_m1 = hidden_error_m1 * relu_deriv_m1

dO1_flat = hidden_error_pre_relu_m1[:, 0:56]
dO2_flat = hidden_error_pre_relu_m1[:, 56:112]
dO3_flat = hidden_error_pre_relu_m1[:, 112:168]
dO4_flat = hidden_error_pre_relu_m1[:, 168:224]
dO5_flat = hidden_error_pre_relu_m1[:, 224:280]

dO1 = dO1_flat.reshape(O1.shape)
dO2 = dO2_flat.reshape(O2.shape)
dO3 = dO3_flat.reshape(O3.shape)
dO4 = dO4_flat.reshape(O4.shape)
dO5 = dO5_flat.reshape(O5.shape)

dH1_1 = np.dot(X.T, dO1)

X2 = X.reshape(N, 2, 392)
dH1_2 = np.tensordot(X2, dO2, axes=([0, 1], [0, 1]))

X3 = X.reshape(N, 7, 112)
dH1_3 = np.tensordot(X3, dO3, axes=([0, 1], [0, 1]))

X4 = X.reshape(N, 14, 56)
dH1_4 = np.tensordot(X4, dO4, axes=([0, 1], [0, 1]))

X5 = X.reshape(N, 28, 28)
dH1_5 = np.tensordot(X5, dO5, axes=([0, 1], [0, 1]))

print("Success! Gradients computed.")
print(f"dH_conv shape: {dH_conv.shape}")
print(f"dH1_5 shape: {dH1_5.shape}")
