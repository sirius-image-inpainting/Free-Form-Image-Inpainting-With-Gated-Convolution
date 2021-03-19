"""
Various modules for SN-PatchGAN generator and discriminator.

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


# ===================== [CODE] =====================


class SpectralConv2d(nn.Module):
    """
    Conv2d with spectral normalization. Consists of vanilla Conv2d
    and spectral normalization for its weights.
    """

    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: Union[Tuple[int], int],
                       stride: Union[Tuple[int], int] = 1,
                       padding: Union[Tuple[int], int] = 0,
                       dilation: Union[Tuple[int], int] = 1,
                       groups: int = 1,
                       bias: bool = True,
                       padding_mode: str = 'zeros'):
        """
        Constructor for SpectralConv2d. For parameter explanation
        see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.
        """

        super(SpectralConv2d, self).__init__()

        self.conv = nn.utils.spectral_norm(nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                            padding_mode=padding_mode))

    def forward(self, X):
        return self.conv(X)

