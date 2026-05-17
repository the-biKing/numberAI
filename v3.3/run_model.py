import sys
import os
import cv2
import numpy as np
import time
import torch
import torch.nn as nn

EMNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                  'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

class MultiScaleSlicing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c2, *h1_weights):
        B = c2.shape[0]
        ctx.save_for_backward(c2, *h1_weights)
        
        o1 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 1, 576), h1_weights[0]).reshape(B, 7, 1, 24)
        o2 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 2, 288), h1_weights[1]).reshape(B, 7, 1, 24)
        o3 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 3, 192), h1_weights[2]).reshape(B, 7, 1, 24)
        o4 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 6, 96),  h1_weights[3]).reshape(B, 7, 1, 24)
        o5 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 8, 72),  h1_weights[4]).reshape(B, 7, 1, 24)
        o6 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 12, 48), h1_weights[5]).reshape(B, 7, 1, 24)
        o7 = torch.einsum('bcid,cdj->bcij', c2.view(B, 7, 24, 24), h1_weights[6]).reshape(B, 7, 1, 24)
        
        oh = []
        for idx in range(4):
            part = c2[:, :, idx*6:(idx+1)*6, :]
            out_p = torch.einsum('bcid,cdj->bcij', part, h1_weights[7+idx]).reshape(B, 7, 1, 24)
            oh.append(out_p)
            
        ov = []
        for idx in range(4):
            part = c2[:, :, :, idx*6:(idx+1)*6]
            out_p = torch.einsum('bcid,cdj->bcij', part, h1_weights[11+idx]).reshape(B, 7, 1, 24)
            ov.append(out_p)
            
        oc = []
        quads = [(0, 12, 0, 12), (0, 12, 12, 24), (12, 24, 0, 12), (12, 24, 12, 24)]
        for idx, (r1, r2, c1, c2_idx) in enumerate(quads):
            part = c2[:, :, r1:r2, c1:c2_idx]
            out_p = torch.einsum('bcid,cdj->bcij', part, h1_weights[15+idx]).reshape(B, 7, 1, 24)
            oc.append(out_p)
            
        return torch.cat([o1, o2, o3, o4, o5, o6, o7] + oh + ov + oc, dim=2)

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        c2 = tensors[0]
        h1_weights = tensors[1:]
        B = c2.shape[0]
        
        grads = torch.chunk(grad_output, 19, dim=2)
        grad_c2 = torch.zeros_like(c2)
        grad_weights = []
        
        # 🐾 修復致命的形狀崩潰 (J = 24 // I)
        configs = [(1, 576), (2, 288), (3, 192), (6, 96), (8, 72), (12, 48), (24, 24)]
        for idx, (I, D) in enumerate(configs):
            J = 24 // I
            g = grads[idx].reshape(B, 7, I, J)
            X = c2.view(B, 7, I, D)
            grad_weights.append(torch.einsum('bcid,bcij->cdj', X, g))
            grad_c2 += torch.einsum('bcij,cdj->bcid', g, h1_weights[idx]).reshape(B, 7, 24, 24)
            
        for idx in range(4):
            g_bcij = grads[7 + idx].reshape(B, 7, 6, 4)
            part = c2[:, :, idx*6:(idx+1)*6, :]
            grad_weights.append(torch.einsum('bcid,bcij->cdj', part, g_bcij))
            grad_c2[:, :, idx*6:(idx+1)*6, :] += torch.einsum('bcij,cdj->bcid', g_bcij, h1_weights[7 + idx])
            
        for idx in range(4):
            g_bcij = grads[11 + idx].reshape(B, 7, 24, 1)
            part = c2[:, :, :, idx*6:(idx+1)*6]
            grad_weights.append(torch.einsum('bcid,bcij->cdj', part, g_bcij))
            grad_c2[:, :, :, idx*6:(idx+1)*6] += torch.einsum('bcij,cdj->bcid', g_bcij, h1_weights[11 + idx])
            
        quads = [(0, 12, 0, 12), (0, 12, 12, 24), (12, 24, 0, 12), (12, 24, 12, 24)]
        for idx, (r1, r2, c1, c2_idx) in enumerate(quads):
            g_bcij = grads[15 + idx].reshape(B, 7, 12, 2)
            part = c2[:, :, r1:r2, c1:c2_idx]
            grad_weights.append(torch.einsum('bcid,bcij->cdj', part, g_bcij))
            grad_c2[:, :, r1:r2, c1:c2_idx] += torch.einsum('bcij,cdj->bcid', g_bcij, h1_weights[15 + idx])
            
        return tuple([grad_c2] + grad_weights)

class ModelV3_1(nn.Module):
    def __init__(self):
        super(ModelV3_1, self).__init__()
        import math
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, padding=0, stride=1, bias=False)
        self.conv2 = nn.Conv2d(7, 7, kernel_size=3, padding=0, stride=1, groups=7, bias=False)
        
        self.H1_1 = nn.Parameter(torch.randn(7, 576, 24) * math.sqrt(2.0 / 576))
        self.H1_2 = nn.Parameter(torch.randn(7, 288, 12) * math.sqrt(2.0 / 288))
        self.H1_3 = nn.Parameter(torch.randn(7, 192, 8) * math.sqrt(2.0 / 192))
        self.H1_4 = nn.Parameter(torch.randn(7, 96, 4) * math.sqrt(2.0 / 96))
        self.H1_5 = nn.Parameter(torch.randn(7, 72, 3) * math.sqrt(2.0 / 72))
        self.H1_6 = nn.Parameter(torch.randn(7, 48, 2) * math.sqrt(2.0 / 48))
        self.H1_7 = nn.Parameter(torch.randn(7, 24, 1) * math.sqrt(2.0 / 24))
        
        self.H1_8 = nn.Parameter(torch.randn(7, 24, 4) * math.sqrt(2.0 / 24))
        self.H1_9 = nn.Parameter(torch.randn(7, 24, 4) * math.sqrt(2.0 / 24))
        self.H1_10 = nn.Parameter(torch.randn(7, 24, 4) * math.sqrt(2.0 / 24))
        self.H1_11 = nn.Parameter(torch.randn(7, 24, 4) * math.sqrt(2.0 / 24))
        
        self.H1_12 = nn.Parameter(torch.randn(7, 6, 1) * math.sqrt(2.0 / 6))
        self.H1_13 = nn.Parameter(torch.randn(7, 6, 1) * math.sqrt(2.0 / 6))
        self.H1_14 = nn.Parameter(torch.randn(7, 6, 1) * math.sqrt(2.0 / 6))
        self.H1_15 = nn.Parameter(torch.randn(7, 6, 1) * math.sqrt(2.0 / 6))
        
        self.H1_16 = nn.Parameter(torch.randn(7, 12, 2) * math.sqrt(2.0 / 12))
        self.H1_17 = nn.Parameter(torch.randn(7, 12, 2) * math.sqrt(2.0 / 12))
        self.H1_18 = nn.Parameter(torch.randn(7, 12, 2) * math.sqrt(2.0 / 12))
        self.H1_19 = nn.Parameter(torch.randn(7, 12, 2) * math.sqrt(2.0 / 12))
        
        # 🐾 瓶頸層 H2: (24, 6)
        self.H2 = nn.Parameter(torch.randn(24, 6) * math.sqrt(2.0 / 24))
        # 🐾 輸出層 H3: 7 * 19 * 6 = 798
        self.H3 = nn.Parameter(torch.randn(798, 47) * math.sqrt(2.0 / 798))
        
    def forward(self, x):
        B = x.shape[0]
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        
        h1_out = MultiScaleSlicing.apply(
            c2, self.H1_1, self.H1_2, self.H1_3, self.H1_4, self.H1_5, self.H1_6, self.H1_7,
            self.H1_8, self.H1_9, self.H1_10, self.H1_11, self.H1_12, self.H1_13, self.H1_14, self.H1_15,
            self.H1_16, self.H1_17, self.H1_18, self.H1_19
        )
        h1_out = torch.relu(h1_out) 
        
        h2_out = torch.matmul(h1_out, self.H2) 
        h2_out = torch.relu(h2_out)
        
        # 🐾 暴力攤平成 798 維度
        h2_flat = h2_out.view(B, 798)
        
        final_out = torch.matmul(h2_flat, self.H3)
        return final_out

def load_weights(model, hidden_layer_dir):
    try:
        model.conv1.weight.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "conv1.txt")).reshape(7, 1, 3, 3), dtype=torch.float32)
        model.conv2.weight.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "conv2.txt")).reshape(7, 1, 3, 3), dtype=torch.float32)
        
        # 🐾 用野獸的嗅覺自動尋找 19 個檔案並還原形狀！
        for i in range(1, 20):
            attr_name = f"H1_{i}"
            tensor_param = getattr(model, attr_name)
            shape = tensor_param.data.shape
            loaded_data = np.loadtxt(os.path.join(hidden_layer_dir, f"{attr_name}.txt"))
            tensor_param.data = torch.tensor(loaded_data.reshape(shape), dtype=torch.float32)
            
        model.H2.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H2.txt")), dtype=torch.float32)
        model.H3.data = torch.tensor(np.loadtxt(os.path.join(hidden_layer_dir, "H3.txt")), dtype=torch.float32)
    except OSError:
        return False
    return True

def run_inference(image_path, model, quiet=False):
    if not os.path.exists(image_path): return None
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return None
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    corners = [float(image[0, 0]), float(image[0, -1]), float(image[-1, 0]), float(image[-1, -1])]
    if sum(corners) / 4.0 > 127: image = 255 - image

    I = image.astype(np.float32) / 255.0
    
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(I).unsqueeze(0).unsqueeze(0)
        out = model(x_tensor)
        predicted_digit = torch.argmax(out).item()

    if not quiet:
        predicted_chars = EMNIST_CLASSES[predicted_digit] if predicted_digit < len(EMNIST_CLASSES) else str(predicted_digit)
        print(f"=> Predicted: {predicted_chars} (Class {predicted_digit})")
    return predicted_digit

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hidden_layer_dir = os.path.join(script_dir, "hiddenLayer")
    model = ModelV3_1()
    
    if not load_weights(model, hidden_layer_dir):
        print("Error: Model weights not found. Run reset_model.py first.")
        sys.exit(1)
        
    if len(sys.argv) > 1:
        run_inference(sys.argv[1], model, quiet=False)
    else:
        npz_path = os.path.join(script_dir, "..", "emnist_28x28.npz")
        data = np.load(npz_path)
        x_test, y_test = data['x_test'], data['y_test']
        indices = np.random.choice(len(x_test), 100, replace=False)
        
        correct = 0
        model.eval()
        with torch.no_grad():
            for i in range(100):
                I_batch = torch.tensor(x_test[indices[i]].reshape(1, 1, 28, 28), dtype=torch.float32)
                if torch.argmax(model(I_batch)).item() == np.argmax(y_test[indices[i]]):
                    correct += 1
        print(f"[EMNIST Test] Accuracy on 100 samples: {correct/100*100:.2f}%")