import torch.nn as nn
import torch

class MiniUNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.enc1 = self.block(3, 64)
        self.enc2 = self.block(64, 128)
        self.bottleneck = self.block(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.block(128, 64)

        self.pool = nn.MaxPool2d(2)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def block(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        s1 = self.enc1(x) # 1, 64, 128, 128
        s2 = self.enc2(self.pool(s1)) # 1, 128, 64, 64
        b = self.bottleneck(self.pool(s2)) # 1, 256, 32, 32

        d2 = self.dec2(torch.cat((self.up2(b), s2), dim=1)) # 1, 128, 64, 64
        d1 = self.dec1(torch.cat((self.up1(d2), s1), dim=1)) # 1, 64, 128, 128

        return self.out(d1) # 1, num_classes, 128, 128