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

    def __init__(self, in_channels: int = 4,
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

        def spectral_conv2d(inp: int, out: int, kern: int, strd: int, pad: int):
            return nn.Sequential(
                    layers.SpectralConv2d(in_channels=inp, out_channels=out,
                        kernel_size=kern, stride=strd, padding=pad),
                    nn.LeakyReLU(negative_slope=leaky_relu_slope),
                )

        # setting up network config
        self.layers = nn.Sequential(
            spectral_conv2d(in_channels, 64, 5, 2, 2),  # layer 1 (5 x 256 x 256)  -> (64 x 128 x 128)
            spectral_conv2d(64, 128, 5, 2, 2),          # layer 2 (64 x 128 x 128) -> (128 x 64 x 64)
            spectral_conv2d(128, 256, 5, 2, 2),         # layer 3 (128 x 64 x 64)  -> (256 x 32 x 32)
            spectral_conv2d(256, 256, 5, 2, 2),         # layer 4 (256 x 32 x 32)  -> (256 x 16 x 16)
            spectral_conv2d(256, 256, 5, 2, 2),         # layer 5 (256 x 16 x 16)  -> (256 x 8 x 8)
            spectral_conv2d(256, 256, 5, 2, 2),         # layer 6 (256 x 8 x 8)    -> (256 x 4 x 4)
            nn.Flatten(),                               # layer 7 (256 x 4 x 4)    -> (4096)
        )


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)

