"""
Running utility of SN-PatchGAN.

Author: Max Pershin
Email: mepershin@gmail.com
Date: 2021-03-22
"""

# ==================== [IMPORT] ====================

import os
import cv2
import torch
import argparse
import numpy as np
import model.gan as gan
import model.data as data_ops
import matplotlib.pyplot as plt


# ===================== [CODE] =====================


def load_image(image_path: str):
    if image_path is not None:
        return torch.from_numpy(cv2.imread(image_path)).permute(2, 0, 1)

    dataloader = data_ops.PlacesDataset('data/valid/')
    random_image_id = np.random.choice(len(dataloader))
    random_image = dataloader[random_image_id][0:3, :, :]
    return random_image



def load_random_mask():
    mask = torch.from_numpy(data_ops.generate_random_mask())
    mask = mask.view(1, *mask.shape)
    return mask



def transform_image(image):
    return image.detach().permute(1, 2, 0) / 255



def run(checkpoint_path: str, image_path: str):
    model = gan.SNPatchGAN()
    model.load_from_checkpoint(checkpoint_path)

    image = load_image(image_path)
    mask = load_random_mask()
    image_and_mask = torch.cat([image, mask], dim=0)

    with torch.no_grad():
        gan_output = model(torch.unsqueeze(image_and_mask, dim=0))[0]
        gan_output = model.finalize_output(gan_output)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(transform_image(image))
    ax[0].set_title('Original image')
    ax[1].imshow(mask[0, :, :])
    ax[1].set_title('Mask')
    ax[2].imshow(transform_image((1 - mask[0, :, :]) * image))
    ax[2].set_title('Masked image')
    ax[3].imshow(gan_output / 255)
    ax[3].set_title('GAN output')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        type=str,
                        metavar='PATH',
                        help='Path to image',
                        required=False)
    parser.add_argument('--checkpoint',
                        type=str,
                        metavar='PATH',
                        help='Path to checkpoint',
                        required=True)

    args = parser.parse_args()
    image_path = args.image
    checkpoint_path = args.checkpoint
    run(checkpoint_path=checkpoint_path, image_path=image_path)

