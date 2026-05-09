import sys
import os
import cv2
import numpy as np

def run_inference(image_path, H1, H2):
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
    print(f"\n--- Output for {os.path.basename(image_path)} ---")
    print(f"=> Predicted Digit: {predicted_digit}")

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

    # If an argument is provided, use that. Otherwise default to all images in verify directory.
    if len(sys.argv) > 1:
        run_inference(sys.argv[1], H1, H2)
    else:
        verify_dir = os.path.join(script_dir, "verify")
        if not os.path.exists(verify_dir):
            print(f"Error: Default directory {verify_dir} not found.")
            sys.exit(1)
            
        print(f"Running inference on all images in {verify_dir}:")
        for filename in os.listdir(verify_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(verify_dir, filename)
                run_inference(img_path, H1, H2)
