import sys
import os
import cv2
import numpy as np
import time
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Add directories to import teacher model
sys.path.append(os.path.join(parent_dir, "v4.0w"))
from model import BudgetWaveMixEMNIST

def load_quantized_teacher(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        quantized_state_dict = checkpoint["quantized_state_dict"]
        scales = checkpoint["scales"]
        buffers = checkpoint["buffers"]
        
        # Dequantize weights and biases
        state_dict = {}
        for name, W_q in quantized_state_dict.items():
            scale = scales[name]
            dequantized = W_q.numpy().astype(np.float32) * scale
            state_dict[name] = torch.tensor(dequantized, dtype=torch.float32)
            
        # Load registered buffers (Haar kernel, running stats, etc.)
        for name, buffer in buffers.items():
            state_dict[name] = buffer
            
        model.load_state_dict(state_dict)
    except OSError as e:
        print(f"Error loading quantized teacher checkpoint: {e}")
        return False
    return True

def run_inference(image_path, model, quiet=False):
    if not os.path.exists(image_path):
        return None

    # Load as raw grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return None
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    corners = [float(image[0, 0]), float(image[0, -1]), float(image[-1, 0]), float(image[-1, -1])]
    corner_avg = sum(corners) / 4.0
    if corner_avg > 127:
        image = 255 - image

    I = image.astype(np.float32) / 255.0
    
    # Note: The teacher model expects raw EMNIST orientation (untransposed).
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(I).unsqueeze(0).unsqueeze(0)
        out = model(x_tensor)
        predicted_digit = torch.argmax(out).item()

    # The teacher EMNIST class labels list
    EMNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                      'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
                      
    predicted_chars = EMNIST_CLASSES[predicted_digit] if predicted_digit < len(EMNIST_CLASSES) else str(predicted_digit)
    
    if not quiet:
        print(f"=> Predicted: {predicted_chars} (Class {predicted_digit})")
    return predicted_digit

if __name__ == "__main__":
    checkpoint_path = os.path.join(script_dir, "hiddenLayer", "model_q4.pth")
    
    model = BudgetWaveMixEMNIST(num_classes=47)
    
    if not load_quantized_teacher(model, checkpoint_path):
        print("Error: Quantized weights not found. Run quantize.py first.")
        sys.exit(1)
        
    print("Loading 4-bit quantized teacher weights and scales...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Teacher Model Size: {total_params:,} parameters (quantized to 4 bits/parameter)")

    if len(sys.argv) > 1:
        t0 = time.perf_counter()
        run_inference(sys.argv[1], model, quiet=False)
        t1 = time.perf_counter()
        print(f"Inference Time: {(t1 - t0) * 1000:.2f} ms")
    else:
        # Check standard EMNIST dataset locations
        npz_path = os.path.join(parent_dir, "mnist", "emnist_28x28.npz")
        if not os.path.exists(npz_path):
            npz_path = os.path.join(parent_dir, "emnist_28x28.npz")
            
        if not os.path.exists(npz_path):
            print(f"Error: {npz_path} not found.")
            sys.exit(1)
            
        print("\nLoading 100 random samples from EMNIST test dataset...")
        data = np.load(npz_path)
        x_test = data['x_test']
        y_test = data['y_test']
        
        num_samples = 100
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        x_sample = x_test[indices].reshape(-1, 28, 28)
        # Note: The teacher model expects raw EMNIST test set (no transposing!)
        y_sample = y_test[indices]
        
        print(f"Evaluating 100 random samples using 4-bit quantized teacher...")
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
        print(f"[EMNIST Test] 4-Bit Quantized Teacher Accuracy: {correct}/{num_samples} ({accuracy:.2f}%)")
        print(f"Total Inference Time for 100 images: {t1 - t0:.4f}s ({(t1 - t0)*1000/num_samples:.2f} ms/image)")
