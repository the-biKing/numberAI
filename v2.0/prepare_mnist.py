import os
import numpy as np
import cv2
import keras
from keras.datasets import mnist

def process_images(images):
    # Normalize to 0-1 and keep 28x28 shape
    processed = images.astype(np.float32) / 255.0
    return processed

def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.shape[0], num_classes))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1.0
    return one_hot

def main():
    print("Downloading and loading MNIST dataset...")
    (x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
    print('train data=', len(x_train_image))
    print('test data=', len(x_test_image))
    
    print("Processing training images (28x28)...")
    x_train_28 = process_images(x_train_image)
    y_train_onehot = to_one_hot(y_train_label)
    
    print("Processing testing images (28x28)...")
    x_test_28 = process_images(x_test_image)
    y_test_onehot = to_one_hot(y_test_label)
    
    # Save the dataset
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mnist")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mnist_28x28.npz")
    
    print(f"Saving processed dataset to {save_path}...")
    np.savez_compressed(save_path, 
                        x_train=x_train_28, y_train=y_train_onehot,
                        x_test=x_test_28, y_test=y_test_onehot)
    print("Done!")

if __name__ == "__main__":
    main()
