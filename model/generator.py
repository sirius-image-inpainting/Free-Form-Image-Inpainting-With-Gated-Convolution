"""
Generator module for SN-PatchGAN for free form
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

import model.layers as ml
import model.utils


# ===================== [CODE] =====================


class SNPatchGANGenerator(nn.Module):
    """
    SN-PatchGAN generator network module.
    """

    def __init__(self, in_channels: int = 4,
                       leaky_relu_slope : float = 0.2):
        """
        SN-PatchGAN generator constructor.

        Parameters
        ----------
        in_channels : int
            Numbers of input channels.
        leaky_relu_slope : float
            Negative slope for LeakyReLU activation function.
        """

        super(SNPatchGANGenerator, self).__init__()


        def gated_conv2d(inp: int, out: int, kern: int, strd: int, pad: int, dil: int = 1):
            """
            Make GatedConv2d layer (followed by activation function)
            with given parameters.

            Parameters
            ----------
            inp : int
                Input channels count.
            out : int
                Output channels count.
            kern : int
                Kernel size.
            strd : int
                Stride.
            pad : int
                Padding.
            dil : int
                Dilation.
            """

            return nn.Sequential(
                    ml.GatedConv2d(in_channels=inp, out_channels=out, kernel_size=kern,
                        stride=strd, padding=pad, dilation=dil),
                    nn.LeakyReLU(negative_slope=leaky_relu_slope),
                )


        def gated_upconv2d(inp: int, out: int, kern: int, strd: int, pad: int):
            """
            Make GatedUpConv2d layer (followed by activation function)
            with given parameters.

            Parameters
            ----------
            inp : int
                Input channels count.
            out : int
                Output channels count.
            kern : int
                Kernel size.
            strd : int
                Stride.
            pad : int
                Padding.
            """

            return nn.Sequential(
                    ml.GatedUpConv2d(in_channels=inp, out_channels=out, kernel_size=kern,
                        stride=strd, padding=pad),
                    nn.LeakyReLU(negative_slope=leaky_relu_slope),
                )


        self.coarse = nn.Sequential(
            gated_conv2d(in_channels, 32, 5, 1, 2),     # layer 01 (4 x 256 x 256)  -> (32 x 256 x 256)
            gated_conv2d(32, 64, 3, 2, 1),              # layer 02 (32 x 256 x 256) -> (64 x 128 x 128)
            gated_conv2d(64, 64, 3, 1, 1),              # layer 03 (64 x 128 x 128) -> (64 x 128 x 128)
            gated_conv2d(64, 128, 3, 2, 1),             # layer 04 (64 x 128 x 128) -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 05 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 06 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 2, dil=2),     # layer 07 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 4, dil=4),     # layer 08 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 8, dil=8),     # layer 09 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 16, dil=16),   # layer 10 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 11 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 12 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_upconv2d(128, 64, 3, 1, 1),           # layer 13 (128 x 64 x 64)  -> (64 x 128 x 128)
            gated_conv2d(64, 64, 3, 1, 1),              # layer 14 (64 x 128 x 128) -> (64 x 128 x 128)
            gated_upconv2d(64, 32, 3, 1, 1),            # layer 15 (64 x 128 x 128) -> (32 x 256 x 256)
            gated_conv2d(32, 16, 3, 1, 1),              # layer 16 (32 x 256 x 256) -> (16 x 256 x 256)
            gated_conv2d(16, 3, 3, 1, 1),               # layer 17 (16 x 256 x 256) -> (3 x 256 x 256)
            #  nn.Tanh(),                                  # layer 18 (3 x 256 x 256)  -> (3 x 256 x 256)
        )

        self.refine_conv = nn.Sequential(
            gated_conv2d(in_channels, 32, 5, 1, 2),     # layer 01 (4 x 256 x 256)  -> (32 x 256 x 256)
            gated_conv2d(32, 32, 3, 2, 1),              # layer 02 (32 x 256 x 256) -> (32 x 128 x 128)
            gated_conv2d(32, 64, 3, 1, 1),              # layer 03 (32 x 128 x 128) -> (64 x 128 x 128)
            gated_conv2d(64, 64, 3, 2, 1),              # layer 04 (64 x 128 x 128) -> (64 x 64 x 64)
            gated_conv2d(64, 128, 3, 1, 1),             # layer 05 (64 x 64 x 64)   -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 06 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 2, dil=2),     # layer 07 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 4, dil=4),     # layer 08 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 8, dil=8),     # layer 09 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 16, dil=16),   # layer 10 (128 x 64 x 64)  -> (128 x 64 x 64)
        )

        self.refine_attention = nn.Sequential(
            gated_conv2d(in_channels, 32, 5, 1, 2),     # layer 01 (4 x 256 x 256)  -> (32 x 256 x 256)
            gated_conv2d(32, 32, 3, 2, 1),              # layer 02 (32 x 256 x 256) -> (32 x 128 x 128)
            gated_conv2d(32, 64, 3, 1, 1),              # layer 03 (32 x 128 x 128) -> (64 x 128 x 128)
            gated_conv2d(64, 128, 3, 2, 1),             # layer 04 (64 x 128 x 128) -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 05 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 06 (128 x 64 x 64)  -> (128 x 64 x 64)
            ml.SelfAttention(in_channels=128),          # layer 07 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 08 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 09 (128 x 64 x 64)  -> (128 x 64 x 64)
        )

        self.refine_tail = nn.Sequential(
            gated_conv2d(256, 128, 3, 1, 1),            # layer 01 (256 x 64 x 64)  -> (128 x 64 x 64)
            gated_conv2d(128, 128, 3, 1, 1),            # layer 02 (128 x 64 x 64)  -> (128 x 64 x 64)
            gated_upconv2d(128, 64, 3, 1, 1),           # layer 03 (128 x 64 x 64)  -> (64 x 128 x 128)
            gated_conv2d(64, 64, 3, 1, 1),              # layer 04 (64 x 128 x 128) -> (64 x 128 x 128)
            gated_upconv2d(64, 32, 3, 1, 1),            # layer 05 (64 x 128 x 128) -> (32 x 256 x 256)
            gated_conv2d(32, 16, 3, 1, 1),              # layer 06 (32 x 256 x 256) -> (16 x 256 x 256)
            gated_conv2d(16, 3, 3, 1, 1),               # layer 07 (16 x 256 x 256) -> (3 x 256 x 256)
            #  nn.Tanh(),                                  # layer 08 (3 x 256 x 256)  -> (3 x 256 x 256)
        )


    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Forward step for generator module.

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

        # masking input images
        masked_images = shaped_images * (1 - shaped_masks)
        input_tensor  = torch.cat([masked_images, shaped_masks], dim=1)

        # step 1: coarse reconstruct
        X_coarse = self.coarse(input_tensor)
        X_coarse = torch.clamp(X_coarse, -1., 1.)
        X_rec_empty = masked_images + X_coarse * shaped_masks

        # step 2: refinement
        X_rec_with_masks = torch.cat([X_rec_empty, shaped_masks], dim=1)
        X_refine_b1 = self.refine_conv(X_rec_with_masks)
        X_refine_b2 = self.refine_attention(X_rec_with_masks)
        X_refine_all = torch.cat([X_refine_b1, X_refine_b2], dim=1)
        X_refine_out = self.refine_tail(X_refine_all)
        X_refine_out = torch.clamp(X_refine_out, -1., 1.)

        # merging refinement with original image
        X_out = X_refine_out * shaped_masks + shaped_images * (1 - shaped_masks)

        # making image out of tensor
        X_out = model.utils.normalize_tensor(X_out, smin=-1, smax=1, tmin=0, tmax=255)
        X_out = X_out.permute(0, 2, 3, 1)
        return X_out

