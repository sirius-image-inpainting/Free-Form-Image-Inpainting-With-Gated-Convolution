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


    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # normalizing images into [-1; 1] range
        images = (images / 255) * 2 - 1

        # masking images
        images = images * (1. - image_masks)
        masked_images_masks = torch.cat([images, image_masks], dim=1)

        # generator step
        gen_images = self.generator(masked_images_masks)
        return gen_images


    def configure_optimizers(self):
        generator_opt = torch.optim.Adam(self.generator.parameters())
        discriminator_opt = torch.optim.Adam(self.discriminator.parameters())
        return (
                {'optimizer': discriminator_opt, 'frequency': 10},
                {'optimizer': generator_opt,     'frequency': 1},
            )


    def training_step(self, batch, batch_idx, optimizer_idx):
        self.unused_(batch_idx)

        # unpacking batch
        images = batch[0]
        masks = batch[1]
        batch_size = batch.shape[0]

        # init losses
        g_loss = model.loss.GeneratorLoss()
        d_loss = model.loss.DiscriminatorLoss()

        # discriminator training step
        if optimizer_idx == 0:
            fake_images = self(images, masks)
            real_images = batch
            all_images = torch.cat([fake_images, batch])

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

        # unpacking batch
        images = batch[0]
        masks = batch[1]
        batch_size = len(batch)

        # init losses
        g_loss = model.loss.GeneratorLoss()
        d_loss = model.loss.DiscriminatorLoss()

        # generator step
        fake_images = self(images, masks)
        all_images = torch.cat([fake_images, images])
        double_masks = torch.cat([masks, masks], dim=0)

        all_output = self.discriminator(all_images, double_masks)
        fake_output = all_output[:batch_size]
        real_output = all_output[batch_size:]

        self.log('d_loss_val', d_loss(real_output, fake_output).item())
        self.log('g_loss_val', g_loss(fake_output).item())

        for image in self.finalize_output(fake_images):
            self.logger.experiment.log_image('gan_out', image)

