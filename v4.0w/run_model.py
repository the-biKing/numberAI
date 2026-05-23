import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

import cv2
import numpy as np
import time
import torch

from model import BudgetWaveMixEMNIST

EMNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                  'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

CHAR_TO_INDEX = {c: i for i, c in enumerate(EMNIST_CLASSES)}
# Add the merged lowercase characters pointing to their uppercase counterparts
MERGED_LOWERCASE = ['c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z']
for c in MERGED_LOWERCASE:
    CHAR_TO_INDEX[c] = CHAR_TO_INDEX[c.upper()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference(image_path, model, quiet=False):
    if not os.path.exists(image_path):
        return None

    # 1. Load as Grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return None

    # 2. MATCH THE WEIGHTS
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # 3. FIX OVERFLOW: Convert to float before math
    corners = [
        float(image[0, 0]), 
        float(image[0, -1]), 
        float(image[-1, 0]), 
        float(image[-1, -1])
    ]
    corner_avg = sum(corners) / 4.0

    # 4. MNIST RULE: Ink must be bright, Background must be dark
    if corner_avg > 127:
        image = 255 - image

    # 5. Normalize
    I = image.astype(np.float32) / 255.0

    # 6. Inference
    I_tensor = torch.from_numpy(I).unsqueeze(0).unsqueeze(0).to(device) # Shape: (1, 1, 28, 28)
    
    with torch.no_grad():
        A = model(I_tensor)
        predicted_digit = torch.argmax(A, dim=1).item()

    predicted_chars = EMNIST_CLASSES[predicted_digit] if predicted_digit < len(EMNIST_CLASSES) else str(predicted_digit)
    
    if not quiet:
        print(f"=> Predicted: {predicted_chars} (Class {predicted_digit})")
    return predicted_digit

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "hiddenLayer", "model.pth")
    
    if not os.path.exists(model_path):
        print("Error: Model weights not found in hiddenLayer/. Run reset_model.py to initialize them.")
        sys.exit(1)

    print("Loading model...")
    model = BudgetWaveMixEMNIST(num_classes=47)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Size: {total_params:,} parameters")

    # If an argument is provided, use that. Otherwise evaluate both datasets.
    if len(sys.argv) > 1:
        t0 = time.perf_counter()
        run_inference(sys.argv[1], model, quiet=False)
        t1 = time.perf_counter()
        print(f"Inference Time: {(t1 - t0) * 1000:.2f} ms")
    else:
        npz_path = os.path.join(script_dir, "..", "mnist", "emnist_28x28.npz")
        if not os.path.exists(npz_path):
            print(f"Error: {npz_path} not found. Please run prepare_emnist.py first.")
            sys.exit(1)
            
        print("\nLoading 100 random samples from EMNIST test dataset...")
        data = np.load(npz_path)
        x_test = data['x_test'].reshape(-1, 1, 28, 28).astype(np.float32)
        y_test = data['y_test']  # One-hot shape (N, 47)
        
        num_samples = 100
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        x_sample = x_test[indices]
        y_sample = y_test[indices]
        
        print(f"Evaluating {num_samples} random samples...")
        correct = 0
        t0 = time.perf_counter()
        
        x_tensor = torch.from_numpy(x_sample).to(device)
        y_labels = np.argmax(y_sample, axis=1)
        
        with torch.no_grad():
            for i in range(num_samples):
                I_batch = x_tensor[i:i+1] # Shape: (1, 1, 28, 28)
                A = model(I_batch)
                predicted_digit = torch.argmax(A, dim=1).item()
                true_label = y_labels[i]
                
                if predicted_digit == true_label:
                    correct += 1
                
        t1 = time.perf_counter()
        
        accuracy = (correct / num_samples) * 100
        print(f"[EMNIST Test Dataset] Accuracy: {correct}/{num_samples} ({accuracy:.2f}%)")
        print(f"[EMNIST Test Dataset] Total Time: {t1 - t0:.4f}s ({(t1 - t0)*1000/num_samples:.2f} ms/image)")
