import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layers=[2, 7, 14, 28]):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.ModuleList()
        prev_layer = 0
        for layer in feature_layers:
            modules = list(vgg.children())[prev_layer:layer]
            self.features.append(nn.Sequential(*modules))
            prev_layer = layer
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for module in self.features:
            x = module(x)
            features.append(x)
        return features

class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=[0.1, 0.3, 0.5, 1.0]):
        super().__init__()
        self.vgg = VGGFeatureExtractor().cuda()
        self.mse = nn.MSELoss().cuda()
        self.layer_weights = layer_weights
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        feats_x = self.vgg(x)
        feats_y = self.vgg(y)
        loss = 0.0
        for w, fx, fy in zip(self.layer_weights, feats_x, feats_y):
            loss += w * self.mse(fx, fy)
        return loss

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, x, y):
        self.kernel_x = self.kernel_x.to(x.device)
        self.kernel_y = self.kernel_y.to(x.device)

        grad_x_x = F.conv2d(x, self.kernel_x, padding=1)
        grad_x_y = F.conv2d(x, self.kernel_y, padding=1)
        grad_y_x = F.conv2d(y, self.kernel_x, padding=1)
        grad_y_y = F.conv2d(y, self.kernel_y, padding=1)

        grad_x = torch.sqrt(grad_x_x.pow(2) + grad_x_y.pow(2) + 1e-6)
        grad_y = torch.sqrt(grad_y_x.pow(2) + grad_y_y.pow(2) + 1e-6)

        return F.l1_loss(grad_x, grad_y)

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss().cuda()
        self.l2 = nn.MSELoss().cuda()
        self.perceptual = PerceptualLoss().cuda()
        self.gradient = GradientLoss().cuda()

    def forward(self, x, y):
        l1 = 5 * self.l1(x, y)
        l2 = 3 * self.l2(x, y)
        perceptual = 2.0 * self.perceptual(x, y)
        gradient = 15 * self.gradient(x, y)
        return l1 + l2 + perceptual + gradient

class StokesLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.epsilon = 1e-3

    def forward(self, I_alpha_pred, I_alpha_gt):
        I0_pred, I45_pred, I90_pred, I135_pred = I_alpha_pred.chunk(4, dim=1)
        S0_pred = (I0_pred + I45_pred + I90_pred + I135_pred) / 2.0
        S1_pred = I0_pred - I90_pred
        S2_pred = I45_pred - I135_pred

        I0_gt, I45_gt, I90_gt, I135_gt = I_alpha_gt.chunk(4, dim=1)
        S0_gt = (I0_gt + I45_gt + I90_gt + I135_gt) / 2.0
        S1_gt = I0_gt - I90_gt
        S2_gt = I45_gt - I135_gt

        loss_S0 = 10 * self.l1(S0_pred, S0_gt)
        loss_S1 = 20 * self.l1(S1_pred, S1_gt)
        loss_S2 = 20 * self.l1(S2_pred, S2_gt)

        aop_pred = 0.5 * torch.atan2(S2_pred + self.epsilon, S1_pred + self.epsilon)
        aop_gt = 0.5 * torch.atan2(S2_gt + self.epsilon, S1_gt + self.epsilon)
        aop_diff = 2 * (aop_pred - aop_gt)
        aop_loss = 50 * torch.mean(1 - torch.cos(aop_diff))

        dop_pred = torch.sqrt(S1_pred**2 + S2_pred**2) / (S0_pred + self.epsilon)
        dop_gt = torch.sqrt(S1_gt**2 + S2_gt**2) / (S0_gt + self.epsilon)
        dop_loss = 20 * self.l1(dop_pred, dop_gt)

        ortho_loss = 5 * (self.l1(I0_pred + I90_pred, I45_pred + I135_pred) +
                         self.l1(I0_gt + I90_gt, I45_gt + I135_gt))

        return loss_S0 + loss_S1 + loss_S2 + aop_loss + dop_loss + ortho_loss

class TotalLoss(nn.Module):
    def __init__(self, stage=3):
        super().__init__()
        self.content_loss = ContentLoss()
        self.stokes_loss = StokesLoss()
        self.stage = stage

    def forward(self, I_guide, I_gt, I_alpha_pred, I_alpha_gt):
        if self.stage == 2:
            angle_loss = sum(self.content_loss(I_alpha_pred[:, i:i+1], I_alpha_gt[:, i:i+1])
                            for i in range(4)) / 4

            return (0.2 * self.content_loss(I_guide, I_gt) +
                    0.3 * angle_loss +
                    1.2 * self.stokes_loss(I_alpha_pred, I_alpha_gt))

        else:
            angle_loss = sum(self.content_loss(I_alpha_pred[:, i:i+1], I_alpha_gt[:, i:i+1])
                            for i in range(4)) / 4

            return (0.05 * self.content_loss(I_guide, I_gt) +
                    0.2 * angle_loss +
                    2.0 * self.stokes_loss(I_alpha_pred, I_alpha_gt))

if __name__ == "__main__":
    batch_size = 8
    height = 256
    width = 256
    I_guide = torch.rand(batch_size, 1, height, width).cuda()
    I_gt = torch.rand(batch_size, 1, height, width).cuda()
    I_alpha_pred = torch.rand(batch_size, 4, height, width).cuda()
    I_alpha_gt = torch.rand(batch_size, 4, height, width).cuda()

    total_loss = TotalLoss(stage=3).cuda()
    loss = total_loss(I_guide, I_gt, I_alpha_pred, I_alpha_gt)
    print("Total Loss:", loss.item())
