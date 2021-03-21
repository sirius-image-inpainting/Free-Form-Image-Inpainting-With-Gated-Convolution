"""
Full model for SN-PatchGAN for free form
image inpainting.

Author: Max Pershin
Email: mepershin@gmail.com
Date: 2021-03-21
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import model.generator
import model.discriminator

# ===================== [CODE] =====================


class SNPatchGAN(pl.LightningModule):
    """
    SN-PatchGAN implementation.
    """

    def __init__(self):
        super(SNPatchGAN, self).__init__()

        # layers
        self.generator = model.generator.SNPatchGANGenerator()
        self.discriminator = model.discriminator.SNPatchGANDiscriminator()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # reading shape
        batch_size = X.shape[0]
        channels = X.shape[1]
        height = X.shape[2]
        width = X.shape[3]

        # dividing images and masks
        images = X[:, 0:3, :, :]
        image_masks = X[:, 3, :, :].view(batch_size, 1, height, width)

        # generator step
        gen_images = self.generator(X)
        gen_images_masks = torch.cat([gen_images, image_masks], dim=1)

        # discriminator step
        out = self.discriminator(gen_images_masks)
        return out


    def training_step(self, batch, batch_idx, optimizer_idx):
        pass


    def configure_optimizers(self):
        pass


    def on_epoch_end(self):
        pass

