# data/dataloader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

def compute_stokes_parameters(I_0, I_45, I_90, I_135):
    S0 = (I_0 + I_45 + I_90 + I_135) / 2.0
    S1 = I_0 - I_90
    S2 = I_45 - I_135
    return torch.stack([S0, S1, S2], dim=0)  # 输出形状 (3, H, W)

class PolarizedImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.blur_folder = os.path.join(data_dir, 'blur')
        self.sharp_folder = os.path.join(data_dir, 'sharp')
        self.image_files = [f for f in os.listdir(self.blur_folder) if f.endswith('.png')]
        assert set(os.listdir(self.sharp_folder)) == set(self.image_files), "模糊和清晰图像文件名不匹配"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        blur_path = os.path.join(self.blur_folder, filename)
        blur_img = Image.open(blur_path).convert('L')
        blur_array = np.array(blur_img, dtype=np.float32) / 255.0  #  [0,1]

        blur0 = blur_array[1::2, 1::2]   # 0°
        blur45 = blur_array[::2, 1::2]   # 45°
        blur90 = blur_array[::2, ::2]    # 90°
        blur135 = blur_array[1::2, ::2]  # 135°

        blur0 = self._to_tensor(blur0)    # (1, H, W)
        blur45 = self._to_tensor(blur45)
        blur90 = self._to_tensor(blur90)
        blur135 = self._to_tensor(blur135)

        stokes_blur = compute_stokes_parameters(
            blur0.squeeze(0), blur45.squeeze(0),
            blur90.squeeze(0), blur135.squeeze(0)
        )  # (3, H, W)

        B_blur = stokes_blur[0].unsqueeze(0)  # (1, H, W)
        S1_blur = stokes_blur[1].unsqueeze(0)
        S2_blur = stokes_blur[2].unsqueeze(0)

        B_polarized = torch.stack([blur0, blur45, blur90, blur135], dim=0).squeeze(1)  # (4, H, W)

        sharp_path = os.path.join(self.sharp_folder, filename)
        sharp_img = Image.open(sharp_path).convert('L')
        sharp_array = np.array(sharp_img, dtype=np.float32) / 255.0  # [0,1]

        sharp0 = sharp_array[1::2, 1::2]
        sharp45 = sharp_array[::2, 1::2]
        sharp90 = sharp_array[::2, ::2]
        sharp135 = sharp_array[1::2, ::2]

        sharp0 = self._to_tensor(sharp0)
        sharp45 = self._to_tensor(sharp45)
        sharp90 = self._to_tensor(sharp90)
        sharp135 = self._to_tensor(sharp135)

        I_sharp = (sharp0 + sharp45 + sharp90 + sharp135) / 4.0  # (1, H, W)

        I_polarized = torch.stack([sharp0, sharp45, sharp90, sharp135], dim=0).squeeze(1)  # (4, H, W)

        stokes_sharp = compute_stokes_parameters(
            sharp0.squeeze(0), sharp45.squeeze(0),
            sharp90.squeeze(0), sharp135.squeeze(0)
        )  # (3, H, W)

        return (B_blur, S1_blur, S2_blur, B_polarized), (I_sharp, I_polarized, stokes_sharp)

    def _to_tensor(self, x):
        x = Image.fromarray((x * 255).astype(np.uint8))
        if self.transform:
            x = self.transform(x)
        return x  # (1, H, W)

def get_dataloader(data_dir, batch_size=16, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # [0,1]
    ])
    dataset = PolarizedImageDataset(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    train_loader = get_dataloader("val", batch_size=16)
    for batch in train_loader:
        (B_blur, S1_blur, S2_blur, B_polarized), (I_sharp, I_polarized, S0_S1_S2) = batch
        print("B_blur:", B_blur.shape)        # (16, 1, 256, 256)
        print("S1_blur:", S1_blur.shape)      # (16, 1, 256, 256)
        print("B_polarized:", B_polarized.shape)  # (16, 4, 256, 256)
        print("I_sharp:", I_sharp.shape)      # (16, 1, 256, 256)
        print("I_polarized:", I_polarized.shape)  # (16, 4, 256, 256)
        print("S0_S1_S2:", S0_S1_S2.shape)    # (16, 3, 256, 256)
        break
