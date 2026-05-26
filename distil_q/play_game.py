import sys
import os
import numpy as np
import time
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Add directories to import student model from v3.1
import importlib.util
spec = importlib.util.spec_from_file_location("run_model_v3_1", os.path.join(parent_dir, "v3.1", "run_model.py"))
run_model_v3_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_model_v3_1)
ModelV3_1 = run_model_v3_1.ModelV3_1
EMNIST_CLASSES = run_model_v3_1.EMNIST_CLASSES
CHAR_TO_INDEX = run_model_v3_1.CHAR_TO_INDEX

# 1. Load Quantized Model Weights
from run_student import load_quantized_weights

model = ModelV3_1()
hidden_layer_dir = os.path.join(script_dir, "hiddenLayer")

if not load_quantized_weights(model, hidden_layer_dir):
    print("Error: Quantized weights not found. Run quantize.py first.")
    sys.exit(1)

model.eval()

# 2. Load EMNIST Dataset
npz_path = os.path.join(parent_dir, "mnist", "emnist_28x28.npz")
if not os.path.exists(npz_path):
    npz_path = os.path.join(parent_dir, "emnist_28x28.npz")

if not os.path.exists(npz_path):
    print(f"Error: Dataset not found at {npz_path}.")
    sys.exit(1)

print("Loading EMNIST dataset for the showdown...")
data = np.load(npz_path)
x_test = data['x_test']
y_test = data['y_test']
print("Dataset loaded successfully!")

# ASCII Rendering Function
def render_ascii(image_flat):
    image = image_flat.reshape(28, 28)
    # EMNIST dataset is already upright on disk. We render it directly without transpose!
    image_display = image
    
    chars = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]
    output = []
    output.append("+" + "-" * 56 + "+")
    for r in range(28):
        row_str = "|"
        for c in range(28):
            val = image_display[r, c]
            char_idx = min(int(val * len(chars)), len(chars) - 1)
            row_str += chars[char_idx] * 2  # print twice to maintain aspect ratio
        row_str += "|"
        output.append(row_str)
    output.append("+" + "-" * 56 + "+")
    return "\n".join(output)

# 3. Game Loop
human_score = 0
model_score = 0
round_num = 1

print("\n" + "=" * 60)
print("              HUMAN vs. DISTIL_Q: THE EMNIST SHOWDOWN")
print("=" * 60)
print("Instructions:")
print("- Look at the character rendered in your terminal.")
print("- Type your guess (0-9, A-Z, a-z) and press Enter.")
print("- EMNIST is case-sensitive for some letters (e.g. A vs a).")
print("- Type 'stop' at any time to finish and view final scores.")
print("=" * 60)

try:
    while True:
        # Select random index
        idx = np.random.randint(0, len(x_test))
        image = x_test[idx]
        true_label = np.argmax(y_test[idx])
        true_char = EMNIST_CLASSES[true_label]
        
        # Display image
        print(f"\n--- ROUND {round_num} ---")
        print(render_ascii(image))
        
        # Get human guess
        human_guess = ""
        while True:
            human_input = input("Your guess (or 'stop' to exit): ").strip()
            if human_input.lower() == 'stop':
                human_guess = 'stop'
                break
            if len(human_input) == 1 and (human_input in CHAR_TO_INDEX):
                human_guess = human_input
                break
            print("Invalid input. Enter a single alphanumeric character (0-9, A-Z, a-z) or 'stop'.")
            
        if human_guess == 'stop':
            break
            
        # Model prediction (requires upright transposed image)
        image_upright = np.transpose(image.reshape(28, 28))
        I_batch = torch.tensor(image_upright.reshape(1, 1, 28, 28), dtype=torch.float32)
        
        with torch.no_grad():
            out = model(I_batch)
            predicted_digit = torch.argmax(out).item()
            
        model_char = EMNIST_CLASSES[predicted_digit]
        
        # Check correctness
        human_correct = (CHAR_TO_INDEX.get(human_guess) == true_label)
        model_correct = (predicted_digit == true_label)
        
        print(f"\nResults:")
        print(f"  - Actual Character : '{true_char}'")
        print(f"  - Your Guess       : '{human_guess}' (Correct: {human_correct})")
        print(f"  - Model's Guess    : '{model_char}' (Correct: {model_correct})")
        
        # Update scores based on competitive showdown rules:
        # - If only one gets it correct, that side gets a point.
        # - If both get it correct, or both get it wrong, no score change.
        if human_correct and not model_correct:
            human_score += 1
            print("🔥 Point to Human!")
        elif model_correct and not human_correct:
            model_score += 1
            print("🤖 Point to distil_q!")
        elif human_correct and model_correct:
            print("🤝 Both got it correct! (No score change)")
        else:
            print("❌ No one got it correct! (No score change)")
            
        print(f"Scores -> Human: {human_score} | distil_q: {model_score}")
        round_num += 1
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nGame interrupted.")

# Final Screen
print("\n" + "=" * 60)
print("                      FINAL SHOWDOWN SCORES")
print("=" * 60)
print(f"  ROUND COMPLETED : {round_num - 1}")
print(f"  HUMAN SCORE     : {human_score}")
print(f"  DISTIL_Q SCORE  : {model_score}")
print("-" * 60)

if human_score > model_score:
    print("🏆 CONGRATULATIONS! Human wins the EMNIST Showdown!")
elif model_score > human_score:
    print("🤖 DEFEAT! The 8-bit quantized distilled model is superior!")
else:
    print("🤝 IT'S A TIE! A perfect balance of biological and artificial intelligence!")
print("=" * 60 + "\n")
