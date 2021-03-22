"""
Various util functions.
"""

# ==================== [IMPORT] ====================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== [CODE] =====================


def normalize_tensor(data: torch.Tensor,
                     smin: float,
                     smax: float,
                     tmin : float,
                     tmax : float) -> torch.Tensor:
    """
    Normalize tensor values from [smin; smax] range
    into [tmin; tmax] range.

    Parameters
    ----------
    image : torch.Tensor
        Input image.
    tmin : float
        Target range minimum.
    tmax : float
        Target range maximum.
    smin : float
        Source range minimum.
    smax : float
        Source range maximum.
    """

    tmin, tmax = min(tmin, tmax), max(tmin, tmax)
    smin, smax = min(smin, smax), max(smin, smax)

    source_len = smax - smin
    target_len = tmax - tmin

    data = (data - smin) / source_len
    data = (data * target_len) + tmin

    return data

