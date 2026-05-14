import numpy as np
import torchvision
import os

print("Downloading EMNIST bymerge...")
dataset_train = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=True, download=True)
dataset_test = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=False, download=True)

x_train = dataset_train.data.numpy()
y_train = dataset_train.targets.numpy()

x_test = dataset_test.data.numpy()
y_test = dataset_test.targets.numpy()

# EMNIST images need to be transposed
# EMNIST format is transposed, so we rotate them back
def process_images(imgs):
    # EMNIST images are rotated 90 degrees and flipped.
    # To fix, we transpose each image.
    imgs = imgs.transpose((0, 2, 1))
    return imgs

x_train = process_images(x_train)
x_test = process_images(x_test)

new_num_classes = 47
print(f"Number of classes: {new_num_classes}")

# One hot encoding function
def to_one_hot(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

y_train_one_hot = to_one_hot(y_train, new_num_classes)
y_test_one_hot = to_one_hot(y_test, new_num_classes)

# Flatten images and normalize
x_train_flat = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

print(f"x_train shape: {x_train_flat.shape}")
print(f"y_train shape: {y_train_one_hot.shape}")

output_path = "emnist_28x28.npz"
np.savez(output_path, x_train=x_train_flat, y_train=y_train_one_hot, x_test=x_test_flat, y_test=y_test_one_hot)

print(f"Saved to {output_path}")

classes = dataset_train.classes
with open("emnist_labels.txt", "w") as f:
    f.write(f"num_classes={new_num_classes}\n")
    for i, c in enumerate(classes):
        f.write(f"{i}: {c}\n")
        print(f"{i}: {c}")
