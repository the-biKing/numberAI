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

def run_inference(image_path, H1_1, H1_2, H1_3, H1_4, H1_5, H2, H_conv, H3, quiet=False):
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
    I_batch = I.reshape(1, 784)
    O1 = np.dot(I_batch, H1_1)
    
    I_2 = I_batch.reshape(1, 2, 392)
    O2 = np.dot(I_2, H1_2)
    
    I_3 = I_batch.reshape(1, 7, 112)
    O3 = np.dot(I_3, H1_3)
    
    I_4 = I_batch.reshape(1, 14, 56)
    O4 = np.dot(I_4, H1_4)
    
    I_5 = I_batch.reshape(1, 28, 28)
    O5 = np.dot(I_5, H1_5)
    
    concat_O = np.concatenate((O1.reshape(1, 56), O2.reshape(1, 56), O3.reshape(1, 56), O4.reshape(1, 56), O5.reshape(1, 56)), axis=1)
    M1 = np.maximum(0, concat_O)
    
    M2_branch1 = np.dot(M1, H2)
    
    X_img = I_batch.reshape(1, 28, 28)
    X_pad = np.pad(X_img, ((0,0), (1,1), (1,1)), mode='constant')
    shape = (1, 14, 14, 3, 3)
    strides = (X_pad.strides[0], X_pad.strides[1]*2, X_pad.strides[2]*2, X_pad.strides[1], X_pad.strides[2])
    X_blocks = np.lib.stride_tricks.as_strided(X_pad, shape=shape, strides=strides)
    
    M2_branch2 = np.tensordot(X_blocks, H_conv, axes=([3, 4], [0, 1]))
    M2_branch2 = M2_branch2.reshape(1, 196)
    
    M2 = M2_branch1 + M2_branch2
    M2_relu = np.maximum(0, M2)
    
    A = np.dot(M2_relu, H3)

    predicted_digit = np.argmax(A)
    predicted_chars = EMNIST_CLASSES[predicted_digit] if predicted_digit < len(EMNIST_CLASSES) else str(predicted_digit)
    
    if not quiet:
        print(f"=> Predicted: {predicted_chars} (Class {predicted_digit})")
    return predicted_digit

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h1_1_path = os.path.join(script_dir, "hiddenLayer", "H1_1.txt")
    h1_2_path = os.path.join(script_dir, "hiddenLayer", "H1_2.txt")
    h1_3_path = os.path.join(script_dir, "hiddenLayer", "H1_3.txt")
    h1_4_path = os.path.join(script_dir, "hiddenLayer", "H1_4.txt")
    h1_5_path = os.path.join(script_dir, "hiddenLayer", "H1_5.txt")
    h2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")
    h_conv_path = os.path.join(script_dir, "hiddenLayer", "H_conv.txt")
    h3_path = os.path.join(script_dir, "hiddenLayer", "H3.txt")
    
    if not os.path.exists(h1_1_path) or not os.path.exists(h2_path):
        print("Error: Model weights not found in hiddenLayer/. Run reset_model.py to initialize them.")
        sys.exit(1)

    print("Loading weights...")
    H1_1 = np.loadtxt(h1_1_path)
    H1_2 = np.loadtxt(h1_2_path)
    H1_3 = np.loadtxt(h1_3_path)
    H1_4 = np.loadtxt(h1_4_path)
    H1_5 = np.loadtxt(h1_5_path)
    H2 = np.loadtxt(h2_path)
    H_conv = np.loadtxt(h_conv_path).reshape(3, 3)
    H3 = np.loadtxt(h3_path)
    
    total_params = H1_1.size + H1_2.size + H1_3.size + H1_4.size + H1_5.size + H2.size + H_conv.size + H3.size
    total_size_kb = (H1_1.nbytes + H1_2.nbytes + H1_3.nbytes + H1_4.nbytes + H1_5.nbytes + H2.nbytes + H_conv.nbytes + H3.nbytes) / 1024.0
    print(f"Model Size: {total_params:,} parameters ({total_size_kb:.2f} KB)")

    # If an argument is provided, use that. Otherwise evaluate both datasets.
    if len(sys.argv) > 1:
        t0 = time.perf_counter()
        run_inference(sys.argv[1], H1_1, H1_2, H1_3, H1_4, H1_5, H2, H_conv, H3, quiet=False)
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
                    predicted = run_inference(img_path, H1_1, H1_2, H1_3, H1_4, H1_5, H2, H_conv, H3, quiet=True)
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
