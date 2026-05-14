import sys
import os
import cv2
import numpy as np
import time

def run_inference(image_path, H1_1, H1_2, H1_3, H1_4, H2, H_conv, H3, quiet=False):
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
    
    I_4 = I_batch.reshape(1, 28, 28)
    O4 = np.dot(I_4, H1_4)
    
    concat_O = np.concatenate((O1.reshape(1, 28), O2.reshape(1, 28), O3.reshape(1, 28), O4.reshape(1, 28)), axis=1)
    M1 = np.maximum(0, concat_O)
    
    M2_branch1 = np.dot(M1, H2)
    
    X_img = I_batch.reshape(1, 28, 28)
    X_blocks = X_img.reshape(1, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4)
    X_conv = X_blocks.reshape(1, 49, 16)
    M2_branch2 = np.dot(X_conv, H_conv).reshape(1, 49)
    
    M2 = M2_branch1 + M2_branch2
    M2_relu = np.maximum(0, M2)
    A = np.dot(M2_relu, H3)

    predicted_digit = np.argmax(A)
    
    if not quiet:
        print(f"=> Predicted: {predicted_digit}")
    return predicted_digit

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h1_1_path = os.path.join(script_dir, "hiddenLayer", "H1_1.txt")
    h1_2_path = os.path.join(script_dir, "hiddenLayer", "H1_2.txt")
    h1_3_path = os.path.join(script_dir, "hiddenLayer", "H1_3.txt")
    h1_4_path = os.path.join(script_dir, "hiddenLayer", "H1_4.txt")
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
    H2 = np.loadtxt(h2_path)
    H_conv = np.loadtxt(h_conv_path).reshape(16, 1)
    H3 = np.loadtxt(h3_path)
    
    total_params = H1_1.size + H1_2.size + H1_3.size + H1_4.size + H2.size + H_conv.size + H3.size
    total_size_kb = (H1_1.nbytes + H1_2.nbytes + H1_3.nbytes + H1_4.nbytes + H2.nbytes + H_conv.nbytes + H3.nbytes) / 1024.0
    print(f"Model Size: {total_params:,} parameters ({total_size_kb:.2f} KB)")

    # If an argument is provided, use that. Otherwise evaluate both datasets.
    if len(sys.argv) > 1:
        t0 = time.perf_counter()
        run_inference(sys.argv[1], H1_1, H1_2, H1_3, H1_4, H2, H_conv, H3, quiet=False)
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
                    predicted = run_inference(img_path, H1_1, H1_2, H1_3, H1_4, H2, H_conv, H3, quiet=True)
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
