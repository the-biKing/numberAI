import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
hidden_layer_dir = os.path.join(script_dir, "hiddenLayer")

if not os.path.exists(hidden_layer_dir):
    os.makedirs(hidden_layer_dir)

# 🐾 根據 fan_in 動態生成最完美的肌肉量 (He Initialization)
def init_and_save(shape, fan_in, filename):
    std = np.sqrt(2.0 / fan_in)
    W = np.random.randn(*shape) * std
    np.savetxt(os.path.join(hidden_layer_dir, filename), W.reshape(-1, W.shape[-1]), fmt='%f')

print("🐾 正在為奇美拉埋下純粹的破壞神經 (Resetting weights)...")
init_and_save((7, 1, 3, 3), 9, "conv1.txt")
init_and_save((7, 1, 3, 3), 9, "conv2.txt")

# 1-7 原始分支 (輸入維度不同，輸出皆為 24)
init_and_save((7, 576, 24), 576, "H1_1.txt")
init_and_save((7, 288, 12), 288, "H1_2.txt")
init_and_save((7, 192, 8),  192, "H1_3.txt")
init_and_save((7, 96, 4),   96,  "H1_4.txt")
init_and_save((7, 72, 3),   72,  "H1_5.txt")
init_and_save((7, 48, 2),   48,  "H1_6.txt")
init_and_save((7, 24, 1),   24,  "H1_7.txt")

# 8-11 橫向分支 (輸入 24)
for i in range(8, 12):
    init_and_save((7, 24, 4), 24, f"H1_{i}.txt")

# 12-15 縱向分支 (輸入 6)
for i in range(12, 16):
    init_and_save((7, 6, 1), 6, f"H1_{i}.txt")

# 16-19 十字象限分支 (輸入 12)
for i in range(16, 20):
    init_and_save((7, 12, 2), 12, f"H1_{i}.txt")

# 🐾 新的 H2 瓶頸層：把 24 維精華壓縮成 6 維
init_and_save((24, 6), 24, "H2.txt")

# 🐾 新的 H3 嘴巴：7通道 * 19視角 * 6維度 = 798
init_and_save((798, 47), 798, "H3.txt")

# ⚠️ 確保舊的殘差閘門徹底被清除
res_path = os.path.join(hidden_layer_dir, "residual.txt")
if os.path.exists(res_path):
    os.remove(res_path)

print("嗷嗚！骨架已重塑完畢！🐾")