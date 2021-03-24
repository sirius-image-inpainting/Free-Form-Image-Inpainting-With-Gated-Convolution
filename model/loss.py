"""
Various losses for GAN training process.
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import model.utils

# ===================== [CODE] =====================


class GeneratorLoss(nn.Module):
    """
    Generator loss for training SN-PatchGAN (GAN hinge loss).
    L = - E_{z in P_z} [D(G(z))]
    """

    def forward(self, X_fake: torch.Tensor) -> torch.Tensor:
        return (-1) * torch.mean(X_fake)



class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss for training SN-PatchGAN (GAN hinge loss).
    L = E_{x in P_data} [ReLU(1 - D(x))] + E_{z in P_z} [ReLU(1 + D(G(z)))]
    """

    def forward(self, X_real: torch.Tensor, X_fake: torch.Tensor) -> torch.Tensor:
        real_loss = torch.mean(F.relu(1. - X_real))
        fake_loss = torch.mean(F.relu(1. + X_fake))
        return real_loss + fake_loss



class ReconLoss(torch.nn.Module):
    """
    L1 loss between original image and reconstructed images.
    """

    def __init__(self):
        super(ReconLoss, self).__init__()
        self.recon_inmask_alpha  = 1 / 40
        self.recon_outmask_alpha = 1 / 40
        self.coarse_inmask_alpha  = 1 / 40
        self.coarse_outmask_alpha = 1 / 40


    def loss(self, images_1: torch.Tensor,
                   images_2: torch.Tensor,
                   masks: torch.Tensor,
                   coef: float) -> torch.Tensor:

        masks_bit_ratio = torch.mean(masks.view(masks.size(0), -1), dim=1)
        masks_bit_ratio = masks_bit_ratio.view(-1, 1, 1, 1)
        masks = torch.unsqueeze(masks, dim=3)

        masked_diff = torch.abs(images_1 - images_2) * masks / masks_bit_ratio
        loss = coef * torch.mean(masked_diff)
        return loss


    def forward(self, images: torch.Tensor,
                      coarse_images: torch.Tensor,
                      recon_images: torch.Tensor,
                      masks: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor
            Torch tensor of shape (B, H, W, 3) containing original images.
        coarse_images : torch.Tensor
            Torch tensor of shape (B, H, W, 3) containing coarse generated images.
        recon_images : torch.Tensor
            Torch tensor of shape (B, H, W, 3) containing reconstructed images.
        masks : torch.Tensor
            Torch tensor of shape (B, H, W) containing masks.
        """

        recon_inmsak_loss  = self.loss(images, recon_images, masks, self.recon_inmask_alpha)
        recon_outmask_loss = self.loss(images, recon_images, 1- masks, self.recon_outmask_alpha)
        coarse_inmask_loss = self.loss(images, coarse_images, masks, self.coarse_inmask_alpha)
        coarse_outmask_loss = self.loss(images, coarse_images, 1 - masks, self.coarse_outmask_alpha)
        return recon_inmsak_loss + recon_outmask_loss + coarse_inmask_loss + coarse_outmask_loss

