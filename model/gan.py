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
        """
        Forward step for SN-PatchGAN.

        Parameters
        ----------
        images : torch.Tensor
            Torch tensor of shape (B, 256, 256, 3), where B is batch size.
        masks : torch.Tensor
            Torch tensor of shape (B, 256, 256), where B is batch size.

        Returns
        -------
        torch.Tensor of shape (B, H, W, 3), each item in [0; 255]
        """

        gen_images = self.generator(images, masks)
        return gen_images


    def configure_optimizers(self):
        generator_opt = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
        discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=0.0004)
        return (
                {'optimizer': discriminator_opt, 'frequency': 5},
                {'optimizer': generator_opt,     'frequency': 1},
            )


    def training_step(self, batch, batch_idx, optimizer_idx):
        self.unused_(batch_idx)

        # unpacking batch
        images = batch[0]
        masks = batch[1]
        batch_size = len(batch)

        # init losses
        g_loss = model.loss.GeneratorLoss()
        d_loss = model.loss.DiscriminatorLoss()
        r_loss = model.loss.ReconLoss()

        # discriminator training step
        if optimizer_idx == 0:
            fake_images, coarse_raw, recon_raw  = self(images, masks)
            all_images = torch.cat([fake_images, images], dim=0)

            double_masks = torch.cat([masks, masks], dim=0)
            all_output = self.discriminator(all_images, double_masks)

            fake_output = all_output[:batch_size]
            real_output = all_output[batch_size:]

            loss = d_loss(real_output, fake_output)
            self.log('d_loss', loss.item())
            return loss

        # generator training step
        if optimizer_idx == 1:
            fake_images, coarse_raw, recon_raw = self(images, masks)
            d_output = self.discriminator(fake_images, masks)

            loss_1 = g_loss(d_output)
            loss_2 = r_loss(images, coarse_raw, recon_raw, masks)
            loss = loss_1

            if np.random.uniform(0, 1) >= 0.1:
                loss += loss_2

            self.log('g_loss', loss_1.item())
            self.log('r_loss', loss_2.item())
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
        fake_images, coarse_raw, recon_raw = self(images, masks)
        all_images = torch.cat([fake_images, images], dim=0)
        double_masks = torch.cat([masks, masks], dim=0)

        # discriminator step
        all_output = self.discriminator(all_images, double_masks)
        fake_output = all_output[:batch_size]
        real_output = all_output[batch_size:]

        self.log('d_loss_val', d_loss(real_output, fake_output).item())
        self.log('g_loss_val', g_loss(fake_output).item())

        for image in fake_images:
            self.logger.experiment.log_image('gan_out', image.cpu() / 255)

