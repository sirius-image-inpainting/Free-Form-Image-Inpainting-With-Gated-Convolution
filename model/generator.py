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
            gated_conv2d(in_channels, 32, 5, 1, 2),     # layer 01 (5 x 256 x 256)  -> (32 x 256 x 256)
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
        )

        self.refine_conv = nn.Sequential(
            gated_conv2d(in_channels, 32, 5, 1, 2),     # layer 01 (5 x 256 x 256)  -> (32 x 256 x 256)
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
            gated_conv2d(in_channels, 32, 5, 1, 2),     # layer 01 (5 x 256 x 256)  -> ()
            gated_conv2d(32, 32, 3, 2, 1),
            gated_conv2d(32, 64, 3, 1, 1),
            gated_conv2d(64, 128, 3, 2, 1),
            gated_conv2d(128, 128, 3, 1, 1),
            gated_conv2d(128, 128, 3, 1, 1),
            ml.SelfAttention(in_channels=128),
            gated_conv2d(128, 128, 3, 1, 1),
            gated_conv2d(128, 128, 3, 1, 1),
        )

        #  self.refine_tail = nn.Sequential(

        #  )


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        #  X = self.coarse(X)
        X_1 = self.refine_attention(X)
        return X_1

