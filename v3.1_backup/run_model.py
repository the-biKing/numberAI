import sys
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn

EMNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                  'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

CHAR_TO_INDEX = {c: i for i, c in enumerate(EMNIST_CLASSES)}
MERGED_LOWERCASE = ['c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z']
for c in MERGED_LOWERCASE:
    CHAR_TO_INDEX[c] = CHAR_TO_INDEX[c.upper()]

SLICES_CONFIG = [
    (1, 576, 24),
    (2, 288, 12),
    (4, 144, 6),
    (8, 72, 3),
    (24, 24, 1),
    (48, 12, 1),
    (96, 6, 1),
    (192, 3, 1),
    (576, 1, 1),
]

class ModelV3_1(nn.Module):
    def __init__(self):
        super(ModelV3_1, self).__init__()
        import math
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, padding=0, stride=1, bias=False)
        self.conv2 = nn.Conv2d(7, 7, kernel_size=3, padding=0, stride=1, groups=7, bias=False)
        
        # H1 shape: (63, 576, 12). Projects each of the 63 sliced channels from 576 to 12.
        self.H1 = nn.Parameter(torch.randn(63, 576, 12) * math.sqrt(2.0 / 576))
        # H2 shape: (756, 128)
        self.H2 = nn.Parameter(torch.randn(756, 128) * math.sqrt(2.0 / 756))
        # H3 shape: (128, 47)
        self.H3 = nn.Parameter(torch.randn(128, 47) * math.sqrt(2.0 / 128))
        
    def forward(self, x):
        B = x.shape[0]
        c1 = self.conv1(x)
        c2 = self.conv2(c1) # (B, 7, 24, 24)
        
        # Generate the 15 dynamic pixel slicing signatures for all 7 channels -> 105 channels
        sliced_channels = []
        for I, D, J in SLICES_CONFIG:
            # Reshape, transpose, and flatten to obtain the distinct multiscale signature
            c2_sliced = c2.view(B, 7, I, D).transpose(2, 3).reshape(B, 7, 576)
            sliced_channels.append(c2_sliced)
            
        # Concatenate along the channel dimension (dim=1) to get shape (B, 105, 576)
        sliced_all = torch.cat(sliced_channels, dim=1)
        
        # Channel-wise linear projection using torch.einsum:
        # sliced_all: (B, 105, 576)
        # self.H1: (105, 576, 12)
        # Output o: (B, 105, 12)
        o = torch.einsum('bci,cij->bcj', sliced_all, self.H1)
        o = torch.relu(o)
        
        # Flatten to shape (B, 756)
        o_flat = o.reshape(B, 756)
        
        # Dense layers H2 and H3
        h2 = torch.matmul(o_flat, self.H2)
        h2 = torch.relu(h2)
        
        final_out = torch.matmul(h2, self.H3)
        return final_out

def load_weights(model, hidden_layer_dir):
    try:
        model.conv1.weight.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "conv1.txt")).reshape(7, 1, 3, 3), dtype=torch.float32)
        model.conv2.weight.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "conv2.txt")).reshape(7, 1, 3, 3), dtype=torch.float32)
        model.H1.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H1.txt")).reshape(63, 576, 12), dtype=torch.float32)
        model.H2.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H2.txt")).reshape(756, 128), dtype=torch.float32)
        model.H3.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H3.txt")).reshape(128, 47), dtype=torch.float32)
    except OSError:
        return False
    return True

def run_inference(image_path, model, quiet=False):
    if not os.path.exists(image_path):
        return None

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return None
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    corners = [float(image[0, 0]), float(image[0, -1]), float(image[-1, 0]), float(image[-1, -1])]
    corner_avg = sum(corners) / 4.0
    if corner_avg > 127:
        image = 255 - image

    I = image.astype(np.float32) / 255.0
    
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(I).unsqueeze(0).unsqueeze(0) # (1, 1, 28, 28)
        out = model(x_tensor)
        predicted_digit = torch.argmax(out).item()

    predicted_chars = EMNIST_CLASSES[predicted_digit] if predicted_digit < len(EMNIST_CLASSES) else str(predicted_digit)
    
    if not quiet:
        print(f"=> Predicted: {predicted_chars} (Class {predicted_digit})")
    return predicted_digit

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hidden_layer_dir = os.path.join(script_dir, "hiddenLayer")
    
    model = ModelV3_1()
    
    if not load_weights(model, hidden_layer_dir):
        print("Error: Model weights not found in hiddenLayer/. Run reset_model.py to initialize them.")
        sys.exit(1)
        
    print("Loading weights...")
    total_params = sum(p.numel() for p in model.parameters())
    total_size_kb = total_params * 4 / 1024.0 # float32 is 4 bytes
    print(f"Model Size: {total_params:,} parameters ({total_size_kb:.2f} KB)")

    if len(sys.argv) > 1:
        t0 = time.perf_counter()
        run_inference(sys.argv[1], model, quiet=False)
        t1 = time.perf_counter()
        print(f"Inference Time: {(t1 - t0) * 1000:.2f} ms")
    else:
        npz_path = os.path.join(script_dir, "..", "emnist_28x28.npz")
        if not os.path.exists(npz_path):
            print(f"Error: {npz_path} not found. Please run prepare_emnist.py first.")
            sys.exit(1)
            
        print("\nLoading 100 random samples from EMNIST test dataset...")
        data = np.load(npz_path)
        x_test = data['x_test']
        y_test = data['y_test']
        
        num_samples = 100
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        x_sample = x_test[indices]
        y_sample = y_test[indices]
        
        print(f"Evaluating {num_samples} random samples...")
        correct = 0
        t0 = time.perf_counter()
        
        model.eval()
        with torch.no_grad():
            for i in range(num_samples):
                I_batch = torch.tensor(x_sample[i].reshape(1, 1, 28, 28), dtype=torch.float32)
                A = model(I_batch)
                predicted_digit = torch.argmax(A).item()
                true_label = np.argmax(y_sample[i])
                if predicted_digit == true_label:
                    correct += 1
                
        t1 = time.perf_counter()
        
        accuracy = (correct / num_samples) * 100
        print(f"[EMNIST Test Dataset] Accuracy: {correct}/{num_samples} ({accuracy:.2f}%)")
        print(f"[EMNIST Test Dataset] Total Time: {t1 - t0:.4f}s ({(t1 - t0)*1000/num_samples:.2f} ms/image)")
