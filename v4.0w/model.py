import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveMixTokenMixer(nn.Module):
    """
    Parameter-free spatial token mixer simulating a 1-level Haar DWT.
    Splits an (B, C, H, W) tensor into (B, C*4, H//2, W//2) losslessly.
    """
    def __init__(self):
        super().__init__()
        # Fixed Haar Wavelet filters
        kernel = torch.tensor([
            [[ 0.5,  0.5], [ 0.5,  0.5]], # LL (Approximation)
            [[ 0.5,  0.5], [-0.5, -0.5]], # LH (Horizontal)
            [[ 0.5, -0.5], [ 0.5, -0.5]], # HL (Vertical)
            [[ 0.5, -0.5], [-0.5,  0.5]]  # HH (Diagonal)
        ], dtype=torch.float32).unsqueeze(1) # shape: (4, 1, 2, 2)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        B, C, H, W = x.shape
        # Apply the fixed filters to each channel independently
        # Repeat shape from (4, 1, 2, 2) to (C*4, 1, 2, 2)
        w = self.kernel.repeat(C, 1, 1, 1)
        out = F.conv2d(x, w, stride=2, padding=0, groups=C)
        return out

class WaveMixBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.mixer = WaveMixTokenMixer()
        
        # Channel MLP acting on the wavelet bands (implemented as 1x1 Convs)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 4, mlp_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_dim, dim * 4, kernel_size=1)
        )
        
        # Re-projection to original dimensions
        self.proj = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # 1. Spatial Mixing via Wavelet Transform
        mixed = self.mixer(x) 
        # 2. Channel Mixing via MLP
        mixed = self.mlp(mixed)
        # 3. Restore Resolution via bilinear upsampling (parameter-free!)
        upsampled = F.interpolate(mixed, scale_factor=2, mode='bilinear', align_corners=False)
        # 4. Project back and add residual connection
        out = x + self.bn(self.proj(upsampled))
        return out

class BudgetWaveMixEMNIST(nn.Module):
    def __init__(self, num_classes=47, hidden_dim=64):
        super().__init__()
        # Initial projection to inject spatial inductive bias
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # Deep backbone of WaveMix blocks (highly parameter efficient)
        self.layer1 = WaveMixBlock(hidden_dim, hidden_dim * 2)
        self.layer2 = WaveMixBlock(hidden_dim, hidden_dim * 2)
        self.layer3 = WaveMixBlock(hidden_dim, hidden_dim * 2)
        
        # Global pooling kills the spatial dimensions completely without flattening to a huge dense layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Tiny linear head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)
