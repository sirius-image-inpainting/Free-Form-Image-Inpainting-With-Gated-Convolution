"""
Discriminator module for SN-PatchGAN for free form
image inpainting.

Author: Max Pershin
Email: mepershin@gmail.com
Date: 2021-03-19
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (
    Union,
    Tuple,
)

import model.layers as layers


# ===================== [CODE] =====================


class SNPatchGANDiscriminator(nn.Module):
    """
    SN-PatchGAN discriminator network module.
    """

    def __init__(self, in_channels: int = 5,
                       leaky_relu_slope: float = 0.2):
        """
        SN-PatchGAN discriminator constructor.

        Parameters
        ----------
        in_channels : int
            How many channels are passed to discriminator input.
        leaky_relu_slope : float
            Negative slope for LeakyReLU activation function.
        """

        super(SNPatchGANDiscriminator, self).__init__()

        # setting up network config
        self.layers = nn.Sequential(
            # layer 1 (5 x 256 x 256) -> (64 x 128 x 128)
            layers.SpectralConv2d(in_channels=in_channels,
                                  out_channels=64,
                                  kernel_size=5,
                                  stride=2,
                                  padding=2),

            nn.LeakyReLU(negative_slope=leaky_relu_slope),

            # layer 2 (64 x 128 x 128) -> (128 x 64 x 64)
            layers.SpectralConv2d(in_channels=64,
                                  out_channels=128,
                                  kernel_size=5,
                                  stride=2,
                                  padding=2),

            nn.LeakyReLU(negative_slope=leaky_relu_slope),

            # layer 3 (128 x 64 x 64) -> (256 x 32 x 32)
            layers.SpectralConv2d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=5,
                                  stride=2,
                                  padding=2),

            nn.LeakyReLU(negative_slope=leaky_relu_slope),

            # layer 4 (256 x 32 x 32) -> (256 x 16 x 16)
            layers.SpectralConv2d(in_channels=256,
                                  out_channels=256,
                                  kernel_size=5,
                                  stride=2,
                                  padding=2),

            nn.LeakyReLU(negative_slope=leaky_relu_slope),

            # layer 5 (256 x 16 x 16) -> (256 x 8 x 8)
            layers.SpectralConv2d(in_channels=256,
                                  out_channels=256,
                                  kernel_size=5,
                                  stride=2,
                                  padding=2),

            nn.LeakyReLU(negative_slope=leaky_relu_slope),

            # layer 6 (256 x 8 x 8) -> (256 x 4 x 4)
            layers.SpectralConv2d(in_channels=256,
                                  out_channels=256,
                                  kernel_size=5,
                                  stride=2,
                                  padding=2),

            nn.LeakyReLU(negative_slope=leaky_relu_slope),

            # layer 7 (256 x 4 x 4) -> (4096)
            nn.Flatten(),
        )


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)

