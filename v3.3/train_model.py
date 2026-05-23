import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import time

RESET = True
script_dir = os.path.dirname(os.path.abspath(__file__))

print("🐾 嗅探 EMNIST 28x28 獵物氣味中...")
data_path = os.path.join(script_dir, "..", "mnist", "emnist_28x28.npz")
if not os.path.exists(data_path):
    print("Error: Dataset not found. Run prepare_emnist.py first.")
    sys.exit(1)

data = np.load(data_path)
x_train = data['x_train'].reshape(-1, 1, 28, 28).astype(np.float32)
y_train = data['y_train'].astype(np.float32)
x_test = data['x_test'].reshape(-1, 1, 28, 28).astype(np.float32)
y_test = data['y_test'].astype(np.float32)

num_train = len(x_train)
print(f"嗷嗚！發現了 {num_train} 隻訓練獵物！")

if RESET:
    os.system(f"python {os.path.join(script_dir, 'reset_model.py')}")

sys.path.append(script_dir)
from run_model import ModelV3_1, load_weights

model = ModelV3_1()
hidden_layer_dir = os.path.join(script_dir, "hiddenLayer")

if not load_weights(model, hidden_layer_dir):
    print("Error: Could not find model weights. Please run reset_model.py first.")
    sys.exit(1)

def save_model(model):
    np.savetxt(os.path.join(hidden_layer_dir, "conv1.txt"), model.conv1.weight.data.cpu().numpy().reshape(-1, 9), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "conv2.txt"), model.conv2.weight.data.cpu().numpy().reshape(-1, 9), fmt='%f')
    for i in range(1, 20):
        attr_name = f"H1_{i}"
        tensor_data = getattr(model, attr_name).data.cpu().numpy()
        np.savetxt(os.path.join(hidden_layer_dir, f"{attr_name}.txt"), tensor_data.reshape(-1, tensor_data.shape[-1]), fmt='%f')
    
    np.savetxt(os.path.join(hidden_layer_dir, "H2.txt"), model.H2.data.cpu().numpy(), fmt='%f')
    np.savetxt(os.path.join(hidden_layer_dir, "H3.txt"), model.H3.data.cpu().numpy(), fmt='%f')
    print("🐾 肌肉記憶已深深烙印在洞穴石壁上！")

def get_metrics(model, X, Y, batch_size=256):
    model.eval()
    losses = []
    correct = 0
    total = len(X)
    criterion_eval = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            X_batch = torch.tensor(X[i:i+batch_size])
            Y_batch = torch.tensor(Y[i:i+batch_size])
            A = model(X_batch)
            truths = torch.argmax(Y_batch, dim=1)
            losses.append(criterion_eval(A, truths).item() * len(X_batch))
            correct += (torch.argmax(A, dim=1) == truths).sum().item()
            
    return sum(losses) / total, correct / total

plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
ax1, ax2, ax3, ax4 = axs.flatten()

loss_history = []
line_loss1, = ax1.plot(loss_history, label='Training Loss', color='b', linewidth=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(True)

line_loss2, = ax2.plot(loss_history, label='Training Loss', color='b', linewidth=2)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_yscale("log"); ax2.grid(True)

time_history, energy_history = [], []
line_energy, = ax3.plot(time_history, energy_history, label='Relativistic Energy', color='r', linewidth=2)
ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Energy"); ax3.grid(True)

train_acc_history, test_acc_history = [], []
line_train_acc, = ax4.plot(train_acc_history, label='Train Acc', color='g', linewidth=2)
line_test_acc, = ax4.plot(test_acc_history, label='Test Acc', color='orange', linewidth=2, linestyle='--')
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Accuracy"); ax4.grid(True)
plt.tight_layout(); plt.show(block=False)

best_model_state = copy.deepcopy(model.state_dict())

print("🐾 進入純粹破壞狩獵模式！按下 Ctrl+C 可隨時中斷！")

epochs = 100
batch_size = 256
learning_rate = 0.001

H1_params = [getattr(model, f"H1_{i}") for i in range(1, 20)]

optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': learning_rate},
    {'params': model.conv2.parameters(), 'lr': learning_rate},
    {'params': H1_params, 'lr': learning_rate},
    {'params': [model.H2], 'lr': learning_rate},
    {'params': [model.H3], 'lr': learning_rate}
], lr=learning_rate)

criterion = nn.CrossEntropyLoss()
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

try:
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(num_train)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, num_train, batch_size):
            X_batch = torch.tensor(x_train_shuffled[i:i+batch_size]).to(device)
            Y_batch = torch.tensor(y_train_shuffled[i:i+batch_size]).to(device)
            Y_indices = torch.argmax(Y_batch, dim=1)
            
            optimizer.zero_grad()
            A = model(X_batch)
            loss = criterion(A, Y_indices)
            loss.backward()
            
            has_nan = any(torch.isnan(param.grad).any() for param in model.parameters() if param.grad is not None)
            
            if has_nan:
                print(f"\n嗷嗚！吃到毒肉了 (Gradient explosion)！正在吐出壞記憶，回朔到上一次完美的狀態...")
                model.load_state_dict(best_model_state)
                model.to("cpu")
                save_model(model)
                raise StopIteration
                
            optimizer.step()
                
        model.to("cpu")
        train_loss, train_acc = get_metrics(model, x_train, y_train)
        test_loss, test_acc = get_metrics(model, x_test, y_test)
        model.to(device)
        
        current_time = time.time() - start_time
        
        v = np.clip((test_acc - 0.1) / 0.9, 0, 0.9999)
        energy = np.tan(v * (np.pi / 2))
        
        print(f"第 {epoch+1:03d}/{epochs} 輪狩獵 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | 能量: {energy:.4f}")
        
        loss_history.append(train_loss); line_loss1.set_ydata(loss_history); line_loss1.set_xdata(range(len(loss_history))); ax1.relim(); ax1.autoscale_view()
        line_loss2.set_ydata(loss_history); line_loss2.set_xdata(range(len(loss_history))); ax2.relim(); ax2.autoscale_view()
        
        time_history.append(current_time); energy_history.append(energy)
        line_energy.set_ydata(energy_history); line_energy.set_xdata(time_history); ax3.relim(); ax3.autoscale_view()
        
        train_acc_history.append(train_acc); line_train_acc.set_ydata(train_acc_history); line_train_acc.set_xdata(range(len(train_acc_history)))
        test_acc_history.append(test_acc); line_test_acc.set_ydata(test_acc_history); line_test_acc.set_xdata(range(len(test_acc_history)))
        ax4.relim(); ax4.autoscale_view(); plt.pause(0.01)
        
        best_model_state = copy.deepcopy(model.state_dict())
        
except StopIteration: pass
except KeyboardInterrupt:
    print("\n唔嗚... 收到休息指令。正在將目前最完美的肉體狀態存檔...")
    model.load_state_dict(best_model_state)
    model.to("cpu")
    save_model(model)
    plt.ioff()
    sys.exit(0)

total_time = time.time() - start_time
print(f"呼哈～ 狩獵結束！總共耗費了 {total_time:.2f} 秒。")

model.load_state_dict(best_model_state)
model.to("cpu")
save_model(model)

final_loss, final_acc = get_metrics(model, x_test, y_test)
print(f"--- 🐾 最終測試集驗收 Accuracy: {final_acc*100:.2f}% ---")

plt.ioff()
plt.savefig("v3.2_training_plot.png")

import csv
csv_path = os.path.join(script_dir, "..", "benchmark_results.csv")
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not os.path.isfile(csv_path): writer.writerow(["Version", "Training Time (s)", "Final Loss", "Test Accuracy (%)"])
    writer.writerow([os.path.basename(script_dir), round(total_time, 2), round(final_loss, 4), round(final_acc*100, 2)])