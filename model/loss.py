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



class ReconLoss(nn.Module):
    """
    L1 loss between original image and reconstructed image.
    """

    def __init__(self, alpha: float = 1.2):
        super(ReconLoss, self).__init__()
        self.alpha = alpha


    def forward(self, real: torch.Tensor, fake: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        normalized_real = model.utils.normalize_tensor(real, 0, 255, -1, 1)
        normalized_fake = model.utils.normalize_tensor(fake, 0, 255, -1, 1)
        shaped_real = normalized_real.permute(0, 3, 1, 2)
        shaped_fake = normalized_fake.permute(0, 3, 1, 2)
        shaped_masks = masks.view(masks.size(0), 1, masks.size(1), masks.size(2))
        masks_mean = shaped_masks.view(masks.size(0), -1).mean(1).view(-1, 1, 1, 1)
        loss = torch.abs(shaped_real - shaped_fake) * shaped_masks / masks_mean
        return self.alpha * torch.mean(loss)

