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
    pS0 = S0
    pS1 = S1
    pS2 = S2
    print("-----------------------load--------------------------")
    print(np.mean(pS0.detach().cpu().numpy()))
    print(np.mean(pS1.detach().cpu().numpy()))
    print(np.mean(pS2.detach().cpu().numpy()))
    print("---------------------------END---------------------------")
    return torch.stack([S0, S1, S2], dim=0)  # (3, H, W)


class PolarizedImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.blur_folder = os.path.join(data_dir, 'blur')
        self.sharp_folder = os.path.join(data_dir, 'sharp')
        self.image_files = [f for f in os.listdir(self.blur_folder) if f.endswith('.png')]
        assert set(os.listdir(self.sharp_folder)) == set(self.image_files), "not same"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        filename = self.image_files[idx]

        blur_path = os.path.join(self.blur_folder, filename)
        blur_image = Image.open(blur_path).convert('L')
        blur_img_array = np.array(blur_image)
        blur0 = Image.fromarray(blur_img_array[1::2, 1::2])  # 0°
        blur45 = Image.fromarray(blur_img_array[::2, 1::2])  # 45°
        blur90 = Image.fromarray(blur_img_array[::2, ::2])   # 90°
        blur135 = Image.fromarray(blur_img_array[1::2, ::2]) # 135°


        if self.transform:
            blur0 = self.transform(blur0)  #  (1, H, W)
            blur45 = self.transform(blur45)
            blur90 = self.transform(blur90)
            blur135 = self.transform(blur135)


        B_blur, S1_blur, S2_blur = compute_stokes_parameters(blur0, blur45, blur90, blur135)

        B_polarized = torch.stack([blur0, blur45, blur90, blur135], dim=0)  #  (4, 1, H, W)

        sharp_path = os.path.join(self.sharp_folder, filename)
        sharp_image = Image.open(sharp_path).convert('L')
        sharp_img_array = np.array(sharp_image)
        sharp0 = Image.fromarray(sharp_img_array[1::2, 1::2])  # 0°
        sharp45 = Image.fromarray(sharp_img_array[::2, 1::2])  # 45°
        sharp90 = Image.fromarray(sharp_img_array[::2, ::2])   # 90°
        sharp135 = Image.fromarray(sharp_img_array[1::2, ::2]) # 135°


        # A = 0.5 * np.arctan2(np.array(sharp0).astype(np.float64) - np.array(sharp90).astype(np.float64), np.array(sharp45).astype(np.float64) - np.array(sharp135).astype(np.float64))
        # A = A.astype(np.float64)
        # print(np.mean(A))


        if self.transform:
            sharp0 = self.transform(sharp0)  # 形状为 (1, H, W)
            sharp45 = self.transform(sharp45)
            sharp90 = self.transform(sharp90)
            sharp135 = self.transform(sharp135)

        I_sharp = (sharp0 + sharp45 + sharp90 + sharp135) / 4.0

        I_polarized = torch.stack([sharp0, sharp45, sharp90, sharp135], dim=0)  # (4, 1, H, W)

        S0_S1_S2 = compute_stokes_parameters(sharp0.squeeze(0), sharp45.squeeze(0), sharp90.squeeze(0), sharp135.squeeze(0))

        B_img0, B_img45, B_img90, B_img135 = [img.squeeze().cpu().numpy() for img in B_polarized]
        Ip_img0, Ip_img45, Ip_img90, Ip_img135 = [img.squeeze().cpu().numpy() for img in I_polarized]

        B_I = (B_img0 + B_img45 + B_img90 + B_img135) / 2
        B_Q = B_img0 - B_img90
        B_U = B_img45 - B_img135

        Ip_I = (Ip_img0 + Ip_img45 + Ip_img90 + Ip_img135) / 2
        Ip_Q = Ip_img0 - Ip_img90
        Ip_U = Ip_img45 - Ip_img135

        B_AOP = 0.5 * np.arctan2(B_U, B_Q)
        Ip_AOP = 0.5 * np.arctan2(Ip_U, Ip_Q)

        print("----------B_polarized------------")
        print(f"Mean AOP: {np.mean(B_AOP)} radians")

        print(f"Central region AOP: {(np.mean(B_AOP[128 - 30:128 + 30]) * 180 / np.pi)}")
        print(f"Mean AOP in degrees: {np.mean(B_AOP) * 180 / np.pi}")

        print("----------I_polarized------------")
        print(f"Mean AOP: {np.mean(Ip_AOP):.4f} radians")
        print(f"Central region AOP: {(np.mean(Ip_AOP[128 - 30:128 + 30]) * 180 / np.pi)}")
        print(f"Mean AOP in degrees: {np.mean(Ip_AOP) * 180 / np.pi:.4f}")

        #
        # sharp0_np = sharp0_unpacked.cpu().numpy()
        # sharp45_np = sharp45_unpacked.cpu().numpy()
        # sharp90_np = sharp90_unpacked.cpu().numpy()
        # sharp135_np = sharp135_unpacked.cpu().numpy()
        #
        # A_unpacked = 0.5 * np.arctan2(sharp0_np - sharp90_np, sharp45_np - sharp135_np)
        #
        # print("Average angle from unpacked I_polarized:", np.mean(A_unpacked))

        return (B_blur, S1_blur, S2_blur, B_polarized), (I_sharp, I_polarized, S0_S1_S2)


def to_float32(x):
    return x.to(dtype=torch.float32)

def get_dataloader(data_dir, batch_size=16, shuffle=True, num_workers=4):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.PILToTensor(),
        transforms.Lambda(to_float32),
    ])
    dataset = PolarizedImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    return dataloader


if __name__ == "__main__":
    train_loader = get_dataloader("val", batch_size=1, shuffle=True)

    for batch in train_loader:
        (B_blur, S1_blur, S2_blur, B_polarized), (I_sharp, I_polarized, S0_S1_S2) = batch
        print("B_blur:", B_blur.shape)          # (16, 1, 256, 256)
        print("S1_blur:", S1_blur.shape)        # (16, 1, 256, 256)
        print("B_polarized:", B_polarized.shape) # (16, 4, 1, 256, 256)
        print("I_sharp:", I_sharp.shape)        # (16, 1, 256, 256)
        print("I_polarized:", I_polarized.shape) # (16, 4, 1, 256, 256)
        print("S0_S1_S2:", S0_S1_S2.shape)      # (16, 3, 1, 256, 256)
        break
