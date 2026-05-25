import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveMixTokenMixer(nn.Module):
    """
    Parameter-free spatial token mixer simulating a 1-level or 2-level Haar DWT.
    - Level 1: Splits an (B, C, H, W) tensor into (B, C*4, H//2, W//2) losslessly.
    - Level 2: Recursively decomposes the LL1 band, then upsamples and concatenates all subbands to (B, C*7, H//2, W//2).
    """
    def __init__(self, levels=1):
        super().__init__()
        self.levels = levels
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
        w = self.kernel.repeat(C, 1, 1, 1)
        
        # Apply level 1 DWT
        out1 = F.conv2d(x, w, stride=2, padding=0, groups=C)
        
        if self.levels == 1:
            return out1
            
        elif self.levels == 2:
            H1, W1 = H // 2, W // 2
            out1_reshaped = out1.view(B, C, 4, H1, W1)
            ll1 = out1_reshaped[:, :, 0] # Shape: (B, C, H1, W1)
            lh1 = out1_reshaped[:, :, 1] # Shape: (B, C, H1, W1)
            hl1 = out1_reshaped[:, :, 2] # Shape: (B, C, H1, W1)
            hh1 = out1_reshaped[:, :, 3] # Shape: (B, C, H1, W1)
            
            # Apply Level 2 DWT to ll1
            w2 = self.kernel.repeat(C, 1, 1, 1)
            out2 = F.conv2d(ll1, w2, stride=2, padding=0, groups=C)
            H2, W2 = H1 // 2, W1 // 2
            out2_reshaped = out2.view(B, C, 4, H2, W2)
            ll2 = out2_reshaped[:, :, 0]
            lh2 = out2_reshaped[:, :, 1]
            hl2 = out2_reshaped[:, :, 2]
            hh2 = out2_reshaped[:, :, 3]
            
            # Bilinearly upsample Level 2 subbands to Level 1 spatial resolution (H1, W1)
            ll2_up = F.interpolate(ll2, size=(H1, W1), mode='bilinear', align_corners=False)
            lh2_up = F.interpolate(lh2, size=(H1, W1), mode='bilinear', align_corners=False)
            hl2_up = F.interpolate(hl2, size=(H1, W1), mode='bilinear', align_corners=False)
            hh2_up = F.interpolate(hh2, size=(H1, W1), mode='bilinear', align_corners=False)
            
            # Concatenate LL2, LH2, HL2, HH2, LH1, HL1, HH1 for each input channel
            # Shape: (B, C * 7, H1, W1)
            stacked = torch.stack([ll2_up, lh2_up, hl2_up, hh2_up, lh1, hl1, hh1], dim=2)
            return stacked.view(B, C * 7, H1, W1)
            
        else:
            raise ValueError(f"Unsupported DWT level: {self.levels}")

class WaveMixBlock(nn.Module):
    def __init__(self, dim, mlp_dim, levels=1):
        super().__init__()
        self.levels = levels
        self.mixer = WaveMixTokenMixer(levels=levels)
        
        in_channels = dim * (4 if levels == 1 else 7)
        # Channel MLP acting on the wavelet bands (implemented as 1x1 Convs)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mlp_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_dim, in_channels, kernel_size=1)
        )
        
        # Re-projection to original dimensions
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        # Learnable local spatial mixing via group/depthwise convolution
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # 1. Spatial Mixing via Wavelet Transform
        mixed = self.mixer(x) 
        # 2. Channel Mixing via MLP
        mixed = self.mlp(mixed)
        # 3. Restore Resolution via bilinear upsampling (parameter-free!)
        upsampled = F.interpolate(mixed, scale_factor=2, mode='bilinear', align_corners=False)
        # 4. Project back
        projected = self.proj(upsampled)
        # 5. Apply learnable depthwise spatial convolution
        spatial_mixed = self.dw_conv(projected)
        # 6. Add residual connection
        out = x + self.bn(spatial_mixed)
        return out

class BudgetWaveMixEMNIST(nn.Module):
    def __init__(self, num_classes=47, hidden_dim=80):
        super().__init__()
        # Initial projection to inject spatial inductive bias
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # Symmetrical Deep backbone of WaveMix blocks (highly parameter efficient)
        self.layer1 = WaveMixBlock(hidden_dim, hidden_dim * 2, levels=1)
        self.layer2 = WaveMixBlock(hidden_dim, hidden_dim * 2, levels=2)
        self.layer3 = WaveMixBlock(hidden_dim, hidden_dim * 2, levels=2)
        self.layer4 = WaveMixBlock(hidden_dim, hidden_dim * 2, levels=2)
        self.layer5 = WaveMixBlock(hidden_dim, hidden_dim * 2, levels=1)
        
        # Global pooling kills the spatial dimensions completely without flattening to a huge dense layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Tiny linear head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)
