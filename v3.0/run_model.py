import sys
import os
import cv2
import numpy as np
import time

EMNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                  'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

CHAR_TO_INDEX = {c: i for i, c in enumerate(EMNIST_CLASSES)}
# Add the merged lowercase characters pointing to their uppercase counterparts
MERGED_LOWERCASE = ['c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z']
for c in MERGED_LOWERCASE:
    CHAR_TO_INDEX[c] = CHAR_TO_INDEX[c.upper()]

def run_inference(image_path, H1, H2, H3, quiet=False):
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
    I = image.flatten().astype(np.float32) / 255.0

    # 6. Inference
    M1 = np.maximum(0, np.dot(I, H1))
    M2 = np.maximum(0, np.dot(M1, H2))
    A = np.dot(M2, H3)

    predicted_digit = np.argmax(A)
    predicted_chars = EMNIST_CLASSES[predicted_digit] if predicted_digit < len(EMNIST_CLASSES) else str(predicted_digit)
    
    if not quiet:
        print(f"=> Predicted: {predicted_chars} (Class {predicted_digit})")
    return predicted_digit

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h1_path = os.path.join(script_dir, "hiddenLayer", "H1.txt")
    h2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")
    h3_path = os.path.join(script_dir, "hiddenLayer", "H3.txt")
    
    if not os.path.exists(h1_path) or not os.path.exists(h2_path) or not os.path.exists(h3_path):
        print("Error: Model weights not found in hiddenLayer/. Run reset_model.py to initialize them.")
        sys.exit(1)

    print("Loading weights...")
    H1 = np.loadtxt(h1_path)
    H2 = np.loadtxt(h2_path)
    H3 = np.loadtxt(h3_path)
    
    total_params = H1.size + H2.size + H3.size
    total_size_kb = (H1.nbytes + H2.nbytes + H3.nbytes) / 1024.0
    print(f"Model Size: {total_params:,} parameters ({total_size_kb:.2f} KB)")

    # If an argument is provided, use that. Otherwise evaluate both datasets.
    if len(sys.argv) > 1:
        t0 = time.perf_counter()
        run_inference(sys.argv[1], H1, H2, H3, quiet=False)
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
                    predicted = run_inference(img_path, H1, H2, H3, quiet=True)
                    if predicted is not None:
                        # Extract true label from filename (e.g., '0.png' -> '0', 'A_1.PNG' -> 'A')
                        true_label_str = filename.split('_')[0].split('.')[0]
                        if true_label_str in CHAR_TO_INDEX:
                            true_label = CHAR_TO_INDEX[true_label_str]
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
