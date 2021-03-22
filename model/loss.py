"""
Various losses for GAN training process.
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


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

