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
import numpy as np

import model.loss
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
        self.unused_ = lambda x: None


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # reading shape
        batch_size = X.shape[0]
        channels = X.shape[1]
        height = X.shape[2]
        width = X.shape[3]

        # dividing images and masks
        images = X[:, 0:3, :, :]
        image_masks = X[:, 3, :, :].view(batch_size, 1, height, width)

        # normalizing images into [-1; 1] range
        images = (images / 255) * 2 - 1

        # masking images
        images = images * (1. - image_masks)
        masked_images_masks = torch.cat([images, image_masks], dim=1)

        # generator step
        gen_images = self.generator(masked_images_masks)
        gen_images_masks = torch.cat([gen_images, image_masks], dim=1)
        return gen_images_masks


    def finalize_output(self, X: torch.Tensor) -> np.ndarray:
        """
        Get image out of model output.
        """

        images = None

        if len(X.shape) not in [3, 4]:
            raise ValueError('X must be tensor with shape (B, C, W, H) or (C, W, H)')

        # batch version
        if len(X.shape) == 4:
            images = X[:, 0:3, :, :]
            images = images.cpu().numpy()
            images = ((images + 1) / 2) * 255
            return np.transpose(images, axes=(0, 2, 3, 1))

        # standalone version
        image = X[0:3, :, :]
        image = image.cpu().numpy()
        image = ((image + 1) / 2) * 255
        return np.transpose(image, axes=(1, 2, 0))


    def configure_optimizers(self):
        generator_opt = torch.optim.Adam(self.generator.parameters())
        discriminator_opt = torch.optim.Adam(self.discriminator.parameters())
        return (
                {'optimizer': discriminator_opt, 'frequency': 10},
                {'optimizer': generator_opt,     'frequency': 1},
            )


    def training_step(self, batch, batch_idx, optimizer_idx):
        self.unused_(batch_idx)

        g_loss = model.loss.GeneratorLoss()
        d_loss = model.loss.DiscriminatorLoss()
        batch_size = batch.shape[0]

        # discriminator training step
        if optimizer_idx == 0:
            fake_images_and_masks = self(batch)
            real_images_and_masks = batch
            all_images_and_masks = torch.cat([fake_images_and_masks, batch])

            all_output = self.discriminator(all_images_and_masks)
            fake_output = all_output[:batch_size]
            real_output = all_output[batch_size:]

            loss = d_loss(real_output, fake_output)
            self.log('d_loss', loss.item())

            return loss

        # generator training step
        if optimizer_idx == 1:
            fake_images_and_masks = self(batch)
            d_output = self.discriminator(fake_images_and_masks)

            loss = g_loss(d_output)
            self.log('g_loss', loss.item())

            return loss


    def validation_step(self, batch, batch_idx):
        self.unused_(batch_idx)

        g_loss = model.loss.GeneratorLoss()
        d_loss = model.loss.DiscriminatorLoss()
        batch_size = batch.shape[0]

        fake_images_and_masks = self(batch)
        all_images_and_masks = torch.cat([fake_images_and_masks, batch])

        all_output = self.discriminator(all_images_and_masks)
        fake_output = all_output[:batch_size]
        real_output = all_output[batch_size:]

        self.log('d_loss_val', d_loss(real_output, fake_output).item())
        self.log('g_loss_val', g_loss(fake_output).item())

        for image in self.finalize_output(fake_images_and_masks):
            self.logger.experiment.log_image('gan_out', image)

