import torch
import torch.nn as nn
import torch.nn.functional as F

class S1Extractor(nn.Module):
    def __init__(self):
        super(S1Extractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)

class S2Extractor(nn.Module):
    def __init__(self):
        super(S2Extractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FusionBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(FusionBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            out = F.relu(out)
            features.append(out)
        return torch.cat(features[1:], dim=1)

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class SGSC(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, fused_channels):
        super(SGSC, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1),
            nn.ReLU()
        )
        self.se_block = SEBlock(fused_channels)
        self.channel_reduction = nn.Conv2d(fused_channels, in_channels, kernel_size=1)

    def forward(self, x, stokes_features):
        residual = self.bottleneck(x)
        fused = torch.cat((x, stokes_features), dim=1)
        out = self.se_block(fused)
        out = self.channel_reduction(out)
        return x + out

class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.sgsc = SGSC(32, 16, 96)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, stokes_features):
        x = self.encoder(x)
        x = self.sgsc(x, stokes_features)
        x = self.decoder(x)
        return x

class UnpolarizedImageEstimator(nn.Module):
    def __init__(self):
        super(UnpolarizedImageEstimator, self).__init__()
        self.S1_extractor = S1Extractor()
        self.S2_extractor = S2Extractor()
        self.fusion_block = FusionBlock(in_channels=128, growth_rate=32, num_layers=2)
        self.downsampling_stokes = DownsamplingBlock(in_channels=64, out_channels=64)
        self.image_feature_extractor = ImageFeatureExtractor()
        self.downsampling_image = DownsamplingBlock(in_channels=64, out_channels=64)
        self.backbone = Backbone()
    def forward(self, B, S1_S2):
        S1 = S1_S2[:, 0:1, :, :]
        S2 = S1_S2[:, 1:2, :, :]

        S1_features = self.S1_extractor(S1)
        S2_features = self.S2_extractor(S2)

        # print("S1", S1_features.shape)
        # print("S2", S2_features.shape)
        stokes_features = torch.cat((S1_features, S2_features), dim=1)
        # print("stokes", stokes_features.shape)
        stokes_features = self.fusion_block(stokes_features)
        # print("stokes", stokes_features.shape)
        stokes_features = self.downsampling_stokes(stokes_features)
        # print("stokes", stokes_features.shape)

        image_features = self.image_feature_extractor(B)
        # print("image", image_features.shape)
        image_features = self.downsampling_image(image_features)
        # print("image", image_features.shape)

        combined_features = torch.cat((stokes_features, image_features), dim=1)
        # print("stokes", stokes_features.shape, "combined", combined_features.shape)
        I_guide = self.backbone(combined_features, stokes_features)
        return I_guide

# test
if __name__ == "__main__":
    B = torch.randn(2, 1, 256, 256)
    S1_S2 = torch.randn(2, 2, 256, 256)

    model = UnpolarizedImageEstimator()
    I_guide = model(B, S1_S2)

    print("I_guide shape:", I_guide.shape)  # torch.Size([2, 1, 256, 256])