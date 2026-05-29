import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Define the model architecture based on the MPU6050 example
class MNISTModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTModel, self).__init__()
        # In example: Conv1d(in=6, out=16, k=3, pad=1)
        # For MNIST (images): Conv2d(in=1, out=16, k=3, pad=1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # In example: MaxPool1d(k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # In example: Conv1d(in=16, out=32, k=3, pad=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # In example: Global Average Pooling (output size 32)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # In example: Dense1(in=32, out=16)
        self.fc1 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        
        # In example: Dense2(in=16, out=num_classes)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1) # Flatten to (batch, 32)
        
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        return x

def main():
    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # standard normalization
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 4. Initialize Model, Loss, Optimizer
    model = MNISTModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 6. Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")
    
    # Save the trained model
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved to mnist_model.pth")

if __name__ == "__main__":
    main()
