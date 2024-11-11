"""py module for implementing the positional interpolation from Meta Ai Research
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class PositionalInterpolation(nn.Module):
    """ module for implementing the Positional Interpolation from Meta Ai,
    main idea is the method of extending the size of 'context window' efficiently for already pretrained model

    interpolate the extended context window range(unseen) into already pretrained(seen) range, using non-integer indexing

    Args:

    Return:

    References:
        https://arxiv.org/abs/2306.15595
    """
    def __init__(self):
        super().__init__()

    def forward(self):
        return
