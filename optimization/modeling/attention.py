import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

def separable_convolution_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
):
    """ func of "1D separable depth-wise convolution", replacing the "global full attention", firstly suggested by "mobileNet"

    Args:
        q: query matrix, shape (batch_size, seq_len, dim_head)
        k: key matrix, shape (batch_size, seq_len, dim_head)
        v: value matrix, shape (batch_size, seq_len, dim_head)

    reference:
        - https://arxiv.org/pdf/1704.04861
        - https://arxiv.org/pdf/2111.00396
    """

    return