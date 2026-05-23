import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
hidden_layer_dir = os.path.join(script_dir, "hiddenLayer")

if not os.path.exists(hidden_layer_dir):
    os.makedirs(hidden_layer_dir)

def init_and_save(shape, fan_in, filename):
    std = np.sqrt(2.0 / fan_in)
    W = np.random.randn(*shape) * std
    np.savetxt(os.path.join(hidden_layer_dir, filename), W.reshape(-1, W.shape[-1]), fmt='%f')

print("Resetting v3.1 PyTorch model weights with 19 Branches...")
init_and_save((7, 1, 3, 3), 9, "conv1.txt")
init_and_save((7, 1, 3, 3), 9, "conv2.txt")

# 原本的 7 條神經
init_and_save((7, 576, 24), 576, "H1_1.txt")
init_and_save((7, 288, 12), 288, "H1_2.txt")
init_and_save((7, 192, 8),  192, "H1_3.txt")
init_and_save((7, 96, 4),   96,  "H1_4.txt")
init_and_save((7, 72, 3),   72,  "H1_5.txt")
init_and_save((7, 48, 2),   48,  "H1_6.txt")
init_and_save((7, 24, 1),   24,  "H1_7.txt")

# 🐾 新增 8-11：橫向切割 (輸入24)
for i in range(8, 12):
    init_and_save((7, 24, 4), 24, f"H1_{i}.txt")

# 🐾 新增 12-15：縱向切割 (輸入6)
for i in range(12, 16):
    init_and_save((7, 6, 1), 6, f"H1_{i}.txt")

# 🐾 新增 16-19：十字象限切割 (輸入12)
for i in range(16, 20):
    init_and_save((7, 12, 2), 12, f"H1_{i}.txt")

init_and_save((24, 16), 24, "H2.txt")

# 🐾 配合 19 個分支升級的最終大腦與殘差閘門
init_and_save((2128, 47), 2128, "H3.txt")
np.savetxt(os.path.join(hidden_layer_dir, "residual.txt"), np.full(2128, 0.5), fmt='%f')

print("Initialized model weights for v3.1 successfully. 🐾")