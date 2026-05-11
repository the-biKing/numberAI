import sys
import os
import cv2
import numpy as np
import time

def conv2d_single(X, K):
    out = np.zeros((26, 26))
    for i in range(26):
        for j in range(26):
            out[i, j] = np.sum(X[i:i+3, j:j+3] * K)
    return out

def maxpool2d_single(C):
    reshaped = C.reshape(13, 2, 13, 2)
    P = reshaped.max(axis=(1, 3))
    return P

def run_inference(image_path, K, H1, H2, quiet=False):
    if not os.path.exists(image_path):
        return None

    # 1. Load as Grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return None

    # 2. MATCH THE WEIGHTS
    # Determine if the model expects 16x16 (256) or 28x28 (784)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # 3. FIX OVERFLOW: Convert to float before math
    # We use .astype(float) so (255+255+255+255) = 1020.0, not 252
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
    image_norm = image.astype(np.float32) / 255.0

    # 6. Inference
    C = conv2d_single(image_norm, K)
    P = maxpool2d_single(C)
    P_flat = P.flatten()
    
    M = np.dot(P_flat, H1)
    M = np.maximum(0, M)
    A = np.dot(M, H2)

    predicted_digit = np.argmax(A)
    
    if not quiet:
        print(f"=> Predicted: {predicted_digit}")
    return predicted_digit

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h1_path = os.path.join(script_dir, "hiddenLayer", "H1.txt")
    h2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")
    k_path = os.path.join(script_dir, "hiddenLayer", "K.txt")
    
    if not os.path.exists(h1_path) or not os.path.exists(h2_path) or not os.path.exists(k_path):
        print("Error: Model weights not found in hiddenLayer/. Run reset_model.py to initialize them.")
        sys.exit(1)

    print("Loading weights...")
    H1 = np.loadtxt(h1_path)
    H2 = np.loadtxt(h2_path)
    K = np.loadtxt(k_path)
    
    total_params = K.size + H1.size + H2.size
    total_size_kb = (K.nbytes + H1.nbytes + H2.nbytes) / 1024.0
    print(f"Model Size: {total_params:,} parameters ({total_size_kb:.2f} KB)")

    # If an argument is provided, use that. Otherwise evaluate both datasets.
    if len(sys.argv) > 1:
        t0 = time.perf_counter()
        run_inference(sys.argv[1], K, H1, H2, quiet=False)
        t1 = time.perf_counter()
        print(f"Inference Time: {(t1 - t0) * 1000:.2f} ms")
    else:
        verify_dir = os.path.join(script_dir, "..", "verify")
        train_dir = os.path.join(script_dir, "..", "inputImage")
        
        def evaluate_directory(directory, dataset_name):
            if not os.path.exists(directory):
                print(f"\nError: Directory {directory} not found.")
                return
            
            print(f"\nEvaluating {dataset_name} dataset in {os.path.basename(directory)}/...")
            correct = 0
            total = 0
            t0 = time.perf_counter()
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(directory, filename)
                    predicted = run_inference(img_path, K, H1, H2, quiet=True)
                    if predicted is not None:
                        # Extract true label from filename (e.g., '0.png' -> 0, '0_1.PNG' -> 0)
                        true_label_str = filename.split('_')[0].split('.')[0]
                        if true_label_str.isdigit():
                            true_label = int(true_label_str)
                            total += 1
                            if predicted == true_label:
                                correct += 1
            t1 = time.perf_counter()
                                
            if total > 0:
                accuracy = (correct / total) * 100
                print(f"[{dataset_name} Dataset] Accuracy: {correct}/{total} ({accuracy:.2f}%)")
                print(f"[{dataset_name} Dataset] Total Time: {t1 - t0:.4f}s ({(t1 - t0)*1000/total:.2f} ms/image)")
            else:
                print(f"[{dataset_name} Dataset] No valid images found.")

        evaluate_directory(train_dir, "Training")
        evaluate_directory(verify_dir, "Verify")
