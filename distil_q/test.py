import sys
import os
import numpy as np

# Set up paths to match your directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Standard EMNIST mapping (first 10 classes are digits 0-9)
EMNIST_CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def render_ascii_raw(image_flat):
    """
    Renders the image exactly as it is stored in the array,
    without any transposing or rotating.
    """
    # Directly reshape without np.transpose
    image = image_flat.reshape(28, 28)
    
    chars = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]
    output = ["+" + "-" * 56 + "+"]
    
    for r in range(28):
        row_str = "|"
        for c in range(28):
            val = image[r, c]
            # Handle both 0.0-1.0 and 0-255 scaling just in case
            if val > 1.0:
                val = val / 255.0
                
            char_idx = min(int(val * len(chars)), len(chars) - 1)
            row_str += chars[char_idx] * 2  # print twice to maintain aspect ratio
        row_str += "|"
        output.append(row_str)
        
    output.append("+" + "-" * 56 + "+")
    return "\n".join(output)

def main():
    # Attempt to locate the dataset just like the original script
    npz_path = os.path.join(parent_dir, "mnist", "emnist_28x28.npz")
    if not os.path.exists(npz_path):
        npz_path = os.path.join(parent_dir, "emnist_28x28.npz")

    if not os.path.exists(npz_path):
        print(f"Error: Dataset not found at {npz_path}.")
        sys.exit(1)

    print(f"Loading EMNIST dataset from {npz_path}...")
    data = np.load(npz_path)
    x_test = data['x_test']
    y_test = data['y_test']
    
    # Keep track of which digits we've already printed
    found_digits = set()
    
    print("\n" + "=" * 60)
    print(" RAW EMNIST ORIENTATION VIEWER (DIGITS 0-9)")
    print("=" * 60)
    
    # Iterate through the dataset and print the first example of each digit
    for idx in range(len(x_test)):
        # Get the true label
        true_label = np.argmax(y_test[idx])
        
        # We only want to print digits 0-9 (labels 0 to 9)
        if true_label < 10 and true_label not in found_digits:
            found_digits.add(true_label)
            
            char_label = EMNIST_CLASSES[true_label]
            image_flat = x_test[idx]
            
            print(f"\n--- First occurrence of Digit: '{char_label}' (Label Index: {true_label}) ---")
            print(render_ascii_raw(image_flat))
            
        # Stop once we've found and printed all 10 digits
        if len(found_digits) == 10:
            break
            
    print("\nFinished printing all 10 digits in their raw orientation.")

if __name__ == "__main__":
    main()