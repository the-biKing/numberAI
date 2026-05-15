import sys
import os
import cv2
import numpy as np
import time

def run_inference(image_path, H1, H1_1, H1_2, H1_3, H1_4, H1_5, H1_6, H2, quiet=False):
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
    O_H1 = np.dot(I_batch, H1)
    
    I_2d = I_batch.reshape(1, 28, 28)
    
    I1_1 = I_2d[:, 0:7, :]
    O1_1 = np.dot(I1_1, H1_1)
    
    I1_2 = I_2d[:, 7:14, :]
    O1_2 = np.dot(I1_2, H1_2)
    
    I1_3 = I_2d[:, 14:21, :]
    O1_3 = np.dot(I1_3, H1_3)
    
    I1_4 = I_2d[:, 21:28, :]
    O1_4 = np.dot(I1_4, H1_4)
    
    I1_5 = I_2d[:, 0:14, 7:21]
    O1_5 = np.dot(I1_5, H1_5)
    
    I1_6 = I_2d[:, 14:28, 7:21]
    O1_6 = np.dot(I1_6, H1_6)
    
    O_H1_flat = O_H1.reshape(1, 28)
    O1_1_flat = O1_1.reshape(1, 28)
    O1_2_flat = O1_2.reshape(1, 28)
    O1_3_flat = O1_3.reshape(1, 28)
    O1_4_flat = O1_4.reshape(1, 28)
    O1_5_flat = O1_5.reshape(1, 28)
    O1_6_flat = O1_6.reshape(1, 28)
    
    concat_O = np.concatenate((O_H1_flat, O1_1_flat, O1_5_flat, O1_2_flat, O1_3_flat, O1_6_flat, O1_4_flat), axis=1)
    M1 = np.maximum(0, concat_O)
    A = np.dot(M1, H2)

    predicted_digit = np.argmax(A)
    
    if not quiet:
        print(f"=> Predicted: {predicted_digit}")
    return predicted_digit

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h1_path = os.path.join(script_dir, "hiddenLayer", "H1.txt")
    h1_1_path = os.path.join(script_dir, "hiddenLayer", "H1_1.txt")
    h1_2_path = os.path.join(script_dir, "hiddenLayer", "H1_2.txt")
    h1_3_path = os.path.join(script_dir, "hiddenLayer", "H1_3.txt")
    h1_4_path = os.path.join(script_dir, "hiddenLayer", "H1_4.txt")
    h1_5_path = os.path.join(script_dir, "hiddenLayer", "H1_5.txt")
    h1_6_path = os.path.join(script_dir, "hiddenLayer", "H1_6.txt")
    h2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")
    
    if not os.path.exists(h1_path) or not os.path.exists(h2_path):
        print("Error: Model weights not found in hiddenLayer/. Run reset_model.py to initialize them.")
        sys.exit(1)

    print("Loading weights...")
    H1 = np.loadtxt(h1_path)
    H1_1 = np.loadtxt(h1_1_path)
    H1_2 = np.loadtxt(h1_2_path)
    H1_3 = np.loadtxt(h1_3_path)
    H1_4 = np.loadtxt(h1_4_path)
    H1_5 = np.loadtxt(h1_5_path)
    H1_6 = np.loadtxt(h1_6_path)
    H2 = np.loadtxt(h2_path)
    
    total_params = H1.size + H1_1.size + H1_2.size + H1_3.size + H1_4.size + H1_5.size + H1_6.size + H2.size
    total_size_kb = (H1.nbytes + H1_1.nbytes + H1_2.nbytes + H1_3.nbytes + H1_4.nbytes + H1_5.nbytes + H1_6.nbytes + H2.nbytes) / 1024.0
    print(f"Model Size: {total_params:,} parameters ({total_size_kb:.2f} KB)")

    # If an argument is provided, use that. Otherwise evaluate both datasets.
    if len(sys.argv) > 1:
        t0 = time.perf_counter()
        run_inference(sys.argv[1], H1, H1_1, H1_2, H1_3, H1_4, H1_5, H1_6, H2, quiet=False)
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
                    predicted = run_inference(img_path, H1, H1_1, H1_2, H1_3, H1_4, H1_5, H1_6, H2, quiet=True)
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
