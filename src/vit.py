import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y)[0]
        y = self.norm2(x)
        x = x + self.mlp(y)
        return x

class ViTSegmenter(nn.Module):
    def __init__(self, num_classes=3, dim=256, depth=6, patch=16, img_size=128):
        super().__init__()
        self.grid = img_size // patch
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.randn(1, self.grid ** 2, dim) * 0.02)
        self.blocks = nn.Sequential(*[TransformerBlock(dim) for _ in range(depth)])
        self.head = nn.Conv2d(dim, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        dim = x.shape[1]
        x = x.flatten(2).transpose(1,2)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = x.transpose(1, 2).reshape(B, dim, self.grid, self.grid)
        x = self.head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x