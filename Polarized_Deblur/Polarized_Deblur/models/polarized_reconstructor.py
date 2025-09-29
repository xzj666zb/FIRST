# model/polarized_reconstructor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class DualAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_att = SEBlock(channel)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        spatial_att = self.spatial_att(x)
        return x * spatial_att


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            DualAttention(channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class PolarizedImageReconstructor(nn.Module):
    def __init__(self, num_blocks=8):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(5, 128, 3, padding=1),
            ResidualBlock(128)
        )
        self.down1 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )
        self.down2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.mid = nn.Sequential(*[ResidualBlock(512) for _ in range(num_blocks)])

        self.up1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512)
        )

        self.up2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.att_gate1 = DualAttention(256)
        self.att_gate2 = DualAttention(128)

        self.final = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, sharp, blur_polar):
        x0 = torch.cat([sharp, blur_polar], dim=1)

        e1 = self.enc1(x0)  # [B,128,H,W]
        d1 = self.down1(e1)  # [B,256,H/2,W/2]
        e2 = self.enc2(d1)  # [B,256,H/2,W/2]
        d2 = self.down2(e2)  # [B,512,H/4,W/4]

        mid = self.mid(d2)  # [B,512,H/4,W/4]

        u1 = self.up1(mid)  # [B,256,H/2,W/2]
        e2_att = self.att_gate1(e2)
        u1 = torch.cat([u1, e2_att], dim=1)  # [B,512,H/2,W/2]
        u1 = self.dec1(u1)  # [B,512,H/2,W/2]

        u2 = self.up2(u1)  # [B,128,H,W]
        e1_att = self.att_gate2(e1)
        u2 = torch.cat([u2, e1_att], dim=1)  # [B,256,H,W]
        u2 = self.dec2(u2)  # [B,256,H,W]

        return self.final(u2)


if __name__ == "__main__":
    model = PolarizedImageReconstructor()
    test_sharp = torch.randn(2, 1, 256, 256)
    test_blur = torch.randn(2, 4, 256, 256)
    output = model(test_sharp, test_blur)
    print(f"in shape: {test_sharp.shape}, {test_blur.shape}")
    print(f"out shape: {output.shape}")  #  torch.Size([2, 4, 256, 256])
