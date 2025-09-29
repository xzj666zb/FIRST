import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unpolarized_estimator import UnpolarizedImageEstimator
from models.polarized_reconstructor import PolarizedImageReconstructor
from models.loss import TotalLoss
from data.dataloader import get_dataloader
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_EPOCHS_POLARIZED, NUM_EPOCHS_FINETUNE, \
    MODEL_SAVE_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, LOG_SAVE_PATH
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torch.nn.functional import mse_loss
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_SAVE_PATH, exist_ok=True)
ssim_fn = ssim(data_range=1.0).to(DEVICE)


def calculate_psnr(mse, max_pixel=1.0):
    return 10 * torch.log10(max_pixel ** 2 / (mse + 1e-10))


def calculate_metrics(pred, target, max_pixel=1.0):
    pred = torch.clamp(pred, 0, 1)
    mse = mse_loss(pred, target)
    psnr = calculate_psnr(mse, max_pixel)
    ssim_value = ssim_fn(pred, target)
    return mse.item(), psnr.item(), ssim_value.item()


def validate(estimator, reconstructor, val_loader, stage):
    estimator.eval()
    if reconstructor is not None:
        reconstructor.eval()

    total_loss = 0.0
    total_mse, total_psnr, total_ssim = 0.0, 0.0, 0.0
    loss_fn = TotalLoss(stage=stage).to(DEVICE)

    with torch.no_grad():
        for batch in val_loader:
            (B_blur, S1_blur, S2_blur, B_polarized), (I_sharp, I_polarized, _) = batch
            B_blur = B_blur.to(DEVICE)
            S1_blur = S1_blur.to(DEVICE)
            S2_blur = S2_blur.to(DEVICE)
            I_sharp = I_sharp.to(DEVICE)

            if stage == 1:
                I_guide = estimator(B_blur, torch.cat([S1_blur, S2_blur], dim=1))
                loss = loss_fn(I_guide, I_sharp, None, None)
                mse, psnr, ssim_value = calculate_metrics(I_guide, I_sharp)
            else:
                B_polarized = B_polarized.to(DEVICE)
                I_polarized = I_polarized.to(DEVICE)
                I_guide = estimator(B_blur, torch.cat([S1_blur, S2_blur], dim=1))
                I_polarized_pred = reconstructor(B_polarized, I_guide)
                loss = loss_fn(I_guide, I_sharp, I_polarized_pred, I_polarized)
                mse, psnr, ssim_value = calculate_metrics(I_polarized_pred, I_polarized)

            # 数值稳定性检查
            if torch.isnan(loss).any():
                raise RuntimeError("NaN loss occurs during validation; please check the data or model.")

            total_loss += loss.item()
            total_mse += mse
            total_psnr += psnr
            total_ssim += ssim_value

    n = len(val_loader)
    return (
        total_loss / n,
        total_mse / n,
        total_psnr / n,
        total_ssim / n
    )


history = {
    'stage1': {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_psnr': [], 'val_ssim': []},
    'stage2': {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_psnr': [], 'val_ssim': []},
    'stage3': {'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_psnr': [], 'val_ssim': []}
}


class GradualWarmupScheduler:
    def __init__(self, optimizer, multiplier, warmup_epoch):
        self.optimizer = optimizer
        self.multiplier = multiplier
        self.warmup_epoch = warmup_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epoch:
            lrs = [base_lr * self.multiplier * (self.epoch / self.warmup_epoch)
                   for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group['lr'] = lr


def train():
    estimator = UnpolarizedImageEstimator().to(DEVICE)
    reconstructor = PolarizedImageReconstructor().to(DEVICE)

    opt_estimator = optim.AdamW(estimator.parameters(), lr=LEARNING_RATE * 0.5, weight_decay=1e-4)
    opt_reconstructor = optim.AdamW(reconstructor.parameters(), lr=LEARNING_RATE * 0.3, weight_decay=1e-4)
    opt_finetune = optim.AdamW([
        {'params': estimator.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': reconstructor.parameters(), 'lr': LEARNING_RATE * 0.1}
    ], weight_decay=1e-5)

    scheduler_estimator = torch.optim.lr_scheduler.StepLR(opt_estimator, step_size=2, gamma=0.8)
    scheduler_reconstructor = torch.optim.lr_scheduler.CosineAnnealingLR(opt_reconstructor, T_max=5)
    warmup_scheduler = GradualWarmupScheduler(opt_finetune, multiplier=0.1, warmup_epoch=3)

    train_loader = get_dataloader(TRAIN_DATA_PATH, BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(VAL_DATA_PATH, BATCH_SIZE, shuffle=False)

    for epoch in range(NUM_EPOCHS):
        estimator.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):


            B_blur, S1_blur, S2_blur, B_polarized = [t.to(DEVICE) for t in batch[0]]
            I_sharp = batch[1][0].to(DEVICE)

            opt_estimator.zero_grad()
            I_guide = estimator(B_blur, torch.cat([S1_blur, S2_blur], dim=1))
            loss = TotalLoss(stage=1)(I_guide, I_sharp, None, None)

            if torch.isnan(loss).any():
                print(f"Epoch {epoch + 1} {batch_idx} NaN,skip")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), max_norm=0.1)  # 严格梯度裁剪
            opt_estimator.step()
            train_loss += loss.item()

        scheduler_estimator.step()
        avg_train = train_loss / len(train_loader)
        val_loss, val_mse, val_psnr, val_ssim = validate(estimator, None, val_loader, stage=1)
        history['stage1']['train_loss'].append(avg_train)
        history['stage1']['val_loss'].append(val_loss)
        history['stage1']['val_mse'].append(val_mse)
        history['stage1']['val_psnr'].append(val_psnr)
        history['stage1']['val_ssim'].append(val_ssim)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | trainloss: {avg_train:.4f} | valloss: {val_loss:.4f} | MSE: {val_mse:.4f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

    torch.save(estimator.state_dict(), os.path.join(MODEL_SAVE_PATH, "unpolarized_estimator.pth"))

    best_val = float('inf')
    for epoch in range(NUM_EPOCHS_POLARIZED):
        estimator.eval()
        reconstructor.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            B_blur, S1_blur, S2_blur, B_polarized = [t.to(DEVICE) for t in batch[0]]
            I_sharp = batch[1][0].to(DEVICE)
            I_polarized = batch[1][1].to(DEVICE)

            opt_reconstructor.zero_grad()
            with torch.no_grad():
                I_guide = estimator(B_blur, torch.cat([S1_blur, S2_blur], dim=1))
            I_pred = reconstructor(B_polarized, I_guide)

            if torch.isnan(loss).any():
                print(f"Epoch {epoch + 1} {batch_idx} NaN,skip")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), max_norm=0.05)
            opt_reconstructor.step()
            train_loss += loss.item()

        scheduler_reconstructor.step()
        avg_train = train_loss / len(train_loader)
        val_loss, val_mse, val_psnr, val_ssim = validate(estimator, reconstructor, val_loader, stage=2)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(reconstructor.state_dict(), os.path.join(MODEL_SAVE_PATH, "polarized_reconstructor.pth"))

        history['stage2']['train_loss'].append(avg_train)
        history['stage2']['val_loss'].append(val_loss)
        history['stage2']['val_mse'].append(val_mse)
        history['stage2']['val_psnr'].append(val_psnr)
        history['stage2']['val_ssim'].append(val_ssim)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS_POLARIZED} | trainloss: {avg_train:.4f} | valloss: {val_loss:.4f} | MSE: {val_mse:.4f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

    for epoch in range(NUM_EPOCHS_FINETUNE):
        estimator.train()
        reconstructor.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            B_blur, S1_blur, S2_blur, B_polarized = [t.to(DEVICE) for t in batch[0]]
            I_sharp = batch[1][0].to(DEVICE)
            I_polarized = batch[1][1].to(DEVICE)

            opt_finetune.zero_grad()
            I_guide = estimator(B_blur, torch.cat([S1_blur, S2_blur], dim=1))
            I_pred = reconstructor(B_polarized, I_guide)

            if torch.isnan(loss).any():
                print(f"Epoch {epoch + 1} {batch_idx}NaN,skip")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(estimator.parameters()) + list(reconstructor.parameters()),
                                           max_norm=0.1)
            opt_finetune.step()
            warmup_scheduler.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        val_loss, val_mse, val_psnr, val_ssim = validate(estimator, reconstructor, val_loader, stage=3)
        history['stage3']['train_loss'].append(avg_train)
        history['stage3']['val_loss'].append(val_loss)
        history['stage3']['val_mse'].append(val_mse)
        history['stage3']['val_psnr'].append(val_psnr)
        history['stage3']['val_ssim'].append(val_ssim)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS_FINETUNE} | trainloss: {avg_train:.4f} | valloss: {val_loss:.4f} | MSE: {val_mse:.4f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

    torch.save(estimator.state_dict(), os.path.join(MODEL_SAVE_PATH, "unpolarized_estimator_final.pth"))
    torch.save(reconstructor.state_dict(), os.path.join(MODEL_SAVE_PATH, "polarized_reconstructor_final.pth"))


if __name__ == "__main__":
    train()