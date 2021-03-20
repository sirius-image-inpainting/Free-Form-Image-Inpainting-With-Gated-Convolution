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


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.conv(X)



class GatedConv2d(nn.Module):
    """
    Gated convolution layer. Consitis of 2 vanilla convolutions (`C_1` and `C_2`)
    and an activation function (`\phi`). Let `X` be input of `GatedConv2d` layer,
    then output is calculated as:

        Gating  = C_1(X)
        Feature = C_2(X)
        Output  = \phi(Feature) * \sigma(Gating)

    where `\sigma` is sigmoid activation function.
    Origin: https://arxiv.org/pdf/1806.03589v2.pdf
    """

    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: Union[Tuple[int], int],
                       stride: Union[Tuple[int], int] = 1,
                       padding: Union[Tuple[int], int] = 0,
                       dilation: Union[Tuple[int], int] = 1,
                       groups: int = 1,
                       bias: bool = True,
                       padding_mode: str = 'zeros',
                       activation: torch.nn.Module = nn.ELU()):
        """
        Constructor for GatedConv2d. For parameter explanation
        see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.

        Parameters
        ----------
        activation : torch.nn.Module
            Feature activation function.
        """

        super(GatedConv2d, self).__init__()

        self.conv_gating = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias,
                                padding_mode=padding_mode)

        self.conv_feature = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias,
                                padding_mode=padding_mode)

        self.gating_act = nn.Sigmoid()
        self.feature_act = activation


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        gating = self.conv_gating(X)
        feature = self.conv_feature(X)
        output = self.feature_act(feature) * self.gating_act(gating)
        return output



class GatedUpConv2d(nn.Module):

    def __init__(self, *args, scale_factor: int = 2, **kwargs):
        """
        Gated convolution layer with scaling. For more information
        see `GatedConv2d` parameter description.

        Parameters
        ----------
        scale_factor : int
            Scaling factor.
        """

        super(GatedUpConv2d, self).__init__()
        self.conv = GatedConv2d(*args, **kwargs)
        self.scaling_factor = scale_factor


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = F.interpolate(X, scale_factor=self.scaling_factor)
        return self.conv(X)



class SelfAttention(nn.Module):

    def __init__(self, in_channels: int,
                       inter_channels: int = None):
        """
        Self attention layer. Originally described at
        https://arxiv.org/pdf/1805.08318.pdf. Authors
        of paper propose `inter_channels` = `in_channels` / 8.
        This logic is implemented when `inter_channels` is None.

        Parameters
        ----------
        in_channels : int
            Channel count in input tensor.
        out_channels : int
            Channel count in  tensor.
        """

        super(SelfAttention, self).__init__()

        if inter_channels is None:
            inter_channels = in_channels // 8

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.gamma = nn.Parameter(torch.zeros(1))

        self.conv_key = nn.Conv2d(in_channels=in_channels,
                                  out_channels=inter_channels,
                                  kernel_size=1)

        self.conv_query = nn.Conv2d(in_channels=in_channels,
                                    out_channels=inter_channels,
                                    kernel_size=1)

        self.conv_value = nn.Conv2d(in_channels=in_channels,
                                    out_channels=inter_channels,
                                    kernel_size=1)

        self.conv_final = nn.Conv2d(in_channels=inter_channels,
                                    out_channels=in_channels,
                                    kernel_size=1)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # saving dimensions
        batch_size = X.shape[0]     # refered as B
        channel_count = X.shape[1]  # refered as C
        height = X.shape[2]         # refered as H
        width = X.shape[3]          # refered as W

        # computing key, query and value
        # (B, c, H, W)
        key = self.conv_key(X)
        query = self.conv_query(X)
        value = self.conv_value(X)

        # resizing matrices (B, c, H, W) -> (B, c, H * W)
        # (`inter_channels` is refered as c)
        key = key.view(batch_size, self.inter_channels, height * width)
        query = query.view(batch_size, self.inter_channels, height * width)
        value = value.view(batch_size, self.inter_channels, height * width)

        # transposing query matrix
        query = query.permute(0, 2, 1) # (B, c, H * W) -> (B, H * W, c)

        # multiplying key and value to get attention scores
        attention = torch.bmm(query, key) # (B, H * W, H * W)

        # normalizing each column of attention score with softmax
        # (column normalization is used not to transpose afterwards)
        attention = torch.softmax(attention, dim=1)

        # multiplying values and attention scores, reshaping
        # (B, c, H * W) x (B, H * W, H * W) -> (B, c, H * W) -> (B, c, H, W)
        att_value = torch.bmm(value, attention)
        att_value = att_value.view(batch_size, self.inter_channels, height, width)

        # going through final convolution
        # (B, c, H, W) -> (B, C, H, W)
        result = self.conv_final(att_value)

        # multiplying with gamma parameter
        result = self.gamma * result + X
        return result

