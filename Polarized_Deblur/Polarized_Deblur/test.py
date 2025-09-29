import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from models.unpolarized_estimator import UnpolarizedImageEstimator
from models.polarized_reconstructor import PolarizedImageReconstructor
from data.dataloader import PolarizedImageDataset, get_dataloader
from config import DEVICE, MODEL_SAVE_PATH, TEST_DATA_PATH

BATCH_SIZE = 1

def test():
    unpolarized_estimator = UnpolarizedImageEstimator().to(DEVICE)
    polarized_reconstructor = PolarizedImageReconstructor().to(DEVICE)

    estimator_path = os.path.join(MODEL_SAVE_PATH, "unpolarized_estimator_final.pth")
    reconstructor_path = os.path.join(MODEL_SAVE_PATH, "polarized_reconstructor_final.pth")
    unpolarized_estimator.load_state_dict(torch.load(estimator_path, weights_only=False))
    polarized_reconstructor.load_state_dict(torch.load(reconstructor_path, weights_only=False))

    unpolarized_estimator.eval()
    polarized_reconstructor.eval()

    test_loader = get_dataloader(TEST_DATA_PATH, batch_size=BATCH_SIZE, shuffle=False)

    output_dir = './polarized_images'
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            print("===================================================")
            (B_blur, S1_blur, S2_blur, B_polarized), (I_sharp, I_polarized, S0_S1_S2) = batch

            B_blur = B_blur.to(DEVICE)
            S1_blur = S1_blur.to(DEVICE)
            S2_blur = S2_blur.to(DEVICE)
            B_polarized = B_polarized.to(DEVICE)
            I_polarized = I_polarized.to(DEVICE)

            # un
            I_guide = unpolarized_estimator(B_blur, torch.cat([S1_blur, S2_blur], dim=1))
            # pol
            I_polarized_pred = polarized_reconstructor(B_polarized.squeeze(2), I_guide).squeeze(0)

            B_polarized_np = B_polarized.squeeze().cpu().numpy().astype(np.float64)
            I_polarized_np = I_polarized.squeeze().cpu().numpy().astype(np.float64)
            I_polarized_pred_np = I_polarized_pred.cpu().numpy().astype(np.float64)
            B_img0, B_img45, B_img90, B_img135 = B_polarized_np
            Ip_img0, Ip_img45, Ip_img90, Ip_img135 = I_polarized_np
            I_img0, I_img45, I_img90, I_img135 = I_polarized_pred_np


            B_I = (B_img0 + B_img45 + B_img90 + B_img135) / 2
            B_Q = B_img0 - B_img90
            B_U = B_img45 - B_img135

            Ip_I = (Ip_img0 + Ip_img45 + Ip_img90 + Ip_img135) / 2
            Ip_Q = Ip_img0 - Ip_img90
            Ip_U = Ip_img45 - Ip_img135

            I_I = (I_img0 + I_img45 + I_img90 + I_img135) / 2
            I_Q = I_img0 - I_img90
            I_U = I_img45 - I_img135

            B_AOP = 0.5 * np.arctan2(B_U, B_Q)
            B_AOP = B_AOP.astype(np.float64)

            Ip_AOP = 0.5 * np.arctan2(Ip_U, Ip_Q)
            Ip_AOP = Ip_AOP.astype(np.float64)

            I_AOP = 0.5 * np.arctan2(I_U, I_Q)
            I_AOP = I_AOP.astype(np.float64)

            # print("B_AOP:", B_AOP)
            # print("I_AOP:", I_AOP)
            print("----------B_polarized------------")
            print(np.mean(B_AOP))
            print(np.mean(B_AOP[128 - 30:128 + 30]) * 180 / np.pi)
            print(np.mean(B_AOP) * 180 / np.pi)

            print("----------I_polarized------------")
            print(np.mean(Ip_AOP))
            print(np.mean(Ip_AOP[128 - 30:128 + 30]) * 180 / np.pi)
            print(np.mean(Ip_AOP) * 180 / np.pi)

            print("----------I_polarized_pred------------")
            print(np.mean(I_AOP))
            print(np.mean(I_AOP[128 - 30:128 + 30]) * 180 / np.pi)
            print(np.mean(I_AOP) * 180 / np.pi)


            # for i in range(I_polarized_pred.size(0)):
            #     single_channel_tensor = I_polarized_pred[i, :, :]
            #
            #     single_channel_tensor = (single_channel_tensor - single_channel_tensor.min()) / (
            #             single_channel_tensor.max() - single_channel_tensor.min())
            #
            #     img_array = single_channel_tensor.cpu().detach().numpy()
            #
            #     img_array = (img_array * 255).astype(np.uint8)
            #
            #     img = Image.fromarray(img_array)
            #
            #     img.save(os.path.join(output_dir, f'angle_{i}_deg.png'))
            #
            # print("Images saved successfully.")


if __name__ == "__main__":
    test()
