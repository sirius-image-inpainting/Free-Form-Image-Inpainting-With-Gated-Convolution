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
import random
import argparse
import numpy as np
import model.gan as gan
import model.data as data_ops
import matplotlib.pyplot as plt


# ===================== [CODE] =====================


def load_image(image_path: str):
    if image_path is not None:
        return torch.from_numpy(cv2.imread(image_path))

    dataloader = data_ops.Dataset('data/valid/')
    random_image = random.choice(dataloader)
    return random_image



def load_random_mask():
    mask = data_ops.generate_random_mask()
    return mask



def run(checkpoint_path: str, image_path: str):
    model = gan.SNPatchGAN()
    model = model.load_from_checkpoint(checkpoint_path)

    origin_image = load_image(image_path)
    origin_mask = load_random_mask()
    channeled_mask = torch.unsqueeze(origin_mask, dim=2)

    with torch.no_grad():
        image = torch.unsqueeze(origin_image, dim=0)
        mask = torch.unsqueeze(origin_mask, dim=0)
        gan_output = model(image, mask)[0][0]

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(origin_image / 255)
    ax[0].set_title('Original image')
    ax[1].imshow(origin_mask)
    ax[1].set_title('Mask')
    ax[2].imshow((1 - channeled_mask) * origin_image / 255)
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

