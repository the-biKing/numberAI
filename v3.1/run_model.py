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

class ModelV3_1(nn.Module):
    def __init__(self):
        super(ModelV3_1, self).__init__()
        import math
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, padding=0, stride=1, bias=False)
        self.conv2 = nn.Conv2d(7, 7, kernel_size=3, padding=0, stride=1, groups=7, bias=False)
        self.H1_1 = nn.Parameter(torch.randn(7, 576, 24) * math.sqrt(2.0 / 576))
        self.H1_2 = nn.Parameter(torch.randn(7, 288, 12) * math.sqrt(2.0 / 288))
        self.H1_3 = nn.Parameter(torch.randn(7, 192, 8) * math.sqrt(2.0 / 192))
        self.H1_4 = nn.Parameter(torch.randn(7, 96, 4) * math.sqrt(2.0 / 96))
        self.H1_5 = nn.Parameter(torch.randn(7, 72, 3) * math.sqrt(2.0 / 72))
        self.H1_6 = nn.Parameter(torch.randn(7, 48, 2) * math.sqrt(2.0 / 48))
        self.H1_7 = nn.Parameter(torch.randn(7, 24, 1) * math.sqrt(2.0 / 24))
        self.H2 = nn.Parameter(torch.randn(24, 16) * math.sqrt(2.0 / 24))
        self.H3 = nn.Parameter(torch.randn(784, 47) * math.sqrt(2.0 / 784))
        
    def forward(self, x):
        B = x.shape[0]
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        
        o1 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 1, 576), self.H1_1).reshape(B, 7, 1, 24)
        o2 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 2, 288), self.H1_2).reshape(B, 7, 1, 24)
        o3 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 3, 192), self.H1_3).reshape(B, 7, 1, 24)
        o4 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 6, 96), self.H1_4).reshape(B, 7, 1, 24)
        o5 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 8, 72), self.H1_5).reshape(B, 7, 1, 24)
        o6 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 12, 48), self.H1_6).reshape(B, 7, 1, 24)
        o7 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 24, 24), self.H1_7).reshape(B, 7, 1, 24)
        
        h1_out = torch.cat([o1, o2, o3, o4, o5, o6, o7], dim=2)
        h1_out = torch.relu(h1_out)
        
        h2_out = torch.matmul(h1_out, self.H2)
        h2_out = torch.relu(h2_out)
        
        concat_res = h2_out.view(B, 784)
        x_flat = x.view(B, 784)
        out = concat_res + x_flat
        
        final_out = torch.matmul(out, self.H3)
        return final_out

def load_weights(model, hidden_layer_dir):
    try:
        model.conv1.weight.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "conv1.txt")).reshape(7, 1, 3, 3), dtype=torch.float32)
        model.conv2.weight.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "conv2.txt")).reshape(7, 1, 3, 3), dtype=torch.float32)
        model.H1_1.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H1_1.txt")).reshape(7, 576, 24), dtype=torch.float32)
        model.H1_2.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H1_2.txt")).reshape(7, 288, 12), dtype=torch.float32)
        model.H1_3.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H1_3.txt")).reshape(7, 192, 8), dtype=torch.float32)
        model.H1_4.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H1_4.txt")).reshape(7, 96, 4), dtype=torch.float32)
        model.H1_5.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H1_5.txt")).reshape(7, 72, 3), dtype=torch.float32)
        model.H1_6.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H1_6.txt")).reshape(7, 48, 2), dtype=torch.float32)
        model.H1_7.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H1_7.txt")).reshape(7, 24, 1), dtype=torch.float32)
        model.H2.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H2.txt")), dtype=torch.float32)
        model.H3.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H3.txt")), dtype=torch.float32)
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
