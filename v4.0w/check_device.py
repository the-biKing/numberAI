import torch
import torch.nn as nn
import time
import sys

print("Diagnosing CUDA speed...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, flush=True)

# Create a small convnet
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(32, 10)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Dummy data
x = torch.randn(256, 1, 28, 28, device=device)
y = torch.randint(0, 10, (256,), device=device)

print("Starting 20 dummy steps...", flush=True)
t0 = time.time()
for i in range(20):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i+1}/20 - Loss: {loss.item():.4f}", flush=True)
t1 = time.time()
print(f"Completed 20 steps in {t1 - t0:.4f} seconds ({(t1 - t0)/20:.4f}s/step)", flush=True)
