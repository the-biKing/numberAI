import cv2
import numpy as np

# Step 1: Load the image using OpenCV
image = cv2.imread("./pyScript/inputImage/3.png")

# Convert the image to RGB (if it's in another format like BGR, as OpenCV uses BGR by default)
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
with open("./pyScript/inputText/3.txt", "w") as file:
    for row in text_representation:
        file.write(row)  # Write each row to a new line in the file

print("Image data has been saved to './pyScript/inputText/3.txt'.")
