import cv2
import numpy as np
import os

# Directories
input_folder = "./pyScript/inputImage/"
output_folder = "./pyScript/inputText/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get a list of all image files in the input folder
image_files = [file for file in os.listdir(input_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

for image_file in image_files:
    # Step 1: Load the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Failed to load image: {image_file}")
        continue

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: Convert the image to text (1 for white, 0 for colored)
    text_representation = []
    height, width, _ = image_rgb.shape

    for y in range(height):  # Loop through each row
        row = []
        for x in range(width):  # Loop through each column
            pixel = image_rgb[y, x]
            # Calculate the sum of RGB values (if all values are near 255, it's white)
            if np.sum(pixel) > 100:  # Check if pixel is white (threshold for near white)
                row.append('0 ')
            else:  # Any other color (colored)
                row.append('1 ')
        text_representation.append(''.join(row))

    # Step 3: Save the text representation to a text file
    output_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
    with open(output_file_path, "w") as file:
        for row in text_representation:
            file.write(row + '\n')  # Write each row to a new line in the file

    print(f"Processed and saved text representation for: {image_file}")