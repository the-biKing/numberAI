import sys
import os
import cv2
import numpy as np
import time

def run_inference(image_path, H1, H2, quiet=False):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape

    # Convert the image to text (1 for colored, 0 for white)
    text_representation = []
    I = [] # Flattened input array
    for y in range(height):
        row = []
        for x in range(width):
            pixel = image_rgb[y, x]
            if np.sum(pixel) > 100:  # Check if pixel is white
                row.append('0 ')
                I.append(0.0)
            else:
                row.append('1 ')
                I.append(1.0)
        text_representation.append(''.join(row))
        
    I = np.array(I)

    # First multiplication
    M = np.dot(H1.T, I)
    
    # Apply ReLU activation function (anything below 0 becomes 0)
    M = np.maximum(0, M)
    
    # Second multiplication
    A = np.dot(H2.T, M)

    predicted_digit = np.argmax(A)
    if not quiet:
        print(f"\n--- Output for {os.path.basename(image_path)} ---")
        print(f"=> Predicted Digit: {predicted_digit}")
    return predicted_digit

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h1_path = os.path.join(script_dir, "hiddenLayer", "H1.txt")
    h2_path = os.path.join(script_dir, "hiddenLayer", "H2.txt")
    
    if not os.path.exists(h1_path) or not os.path.exists(h2_path):
        print("Error: Model weights not found in hiddenLayer/. Run reset_model.py to initialize them.")
        sys.exit(1)

    print("Loading weights...")
    H1 = np.loadtxt(h1_path)
    H2 = np.loadtxt(h2_path)
    
    total_params = H1.size + H2.size
    total_size_kb = (H1.nbytes + H2.nbytes) / 1024.0
    print(f"Model Size: {total_params:,} parameters ({total_size_kb:.2f} KB)")

    # If an argument is provided, use that. Otherwise evaluate both datasets.
    if len(sys.argv) > 1:
        t0 = time.perf_counter()
        run_inference(sys.argv[1], H1, H2, quiet=False)
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
                    predicted = run_inference(img_path, H1, H2, quiet=True)
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
