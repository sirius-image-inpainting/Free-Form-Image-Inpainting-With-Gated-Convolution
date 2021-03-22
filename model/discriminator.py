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
import model.utils


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


    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Forward step for discriminator module.

        Parameters
        ----------
        images : torch.Tensor
            Torch tensor of shape (B, 256, 256, 3), where B is batch size.
        masks : torch.Tensor
            Torch tensor of shape (B, 256, 256), where B is batch size.
        """

        # shaping input data
        normalized_images = model.utils.normalize_tensor(images,
                                    smin=0, smax=255, tmin=-1, tmax=1)
        shaped_images = normalized_images.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        shaped_masks  = torch.unsqueeze(masks, dim=1)          # (B, H, W) -> (B, 1, H, W)

        # evaling discriminator
        input_tensor = torch.cat([shaped_images, shaped_masks], dim=1)
        output = self.layers(input_tensor)
        return output

