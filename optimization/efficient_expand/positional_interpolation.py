"""py module for implementing the positional interpolation from Meta Ai Research
"""
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor


class PositionalInterpolation(nn.Embedding):
    """ module for implementing the Positional Interpolation from Meta Ai,
    main idea is the method of extending the size of 'context window' efficiently for already pretrained model

    interpolate the extended context window range(unseen) into already pretrained(seen) range,
    using non-integer indexing

    main idea:
        f'(x,m) = f(x, mL/L')

    Args:
        extended_length (int): target value of extended context window size
        pretrained_length (int): value of pretrained model's context window size
        dim_head (int): value of size of pretrained model's hidden state

    References:
        https://arxiv.org/abs/2306.15595
    """
    def __init__(self, extended_length: int, pretrained_length: int, dim_head: int) -> None:
        super().__init__(extended_length, dim_head)
        self.scaler = pretrained_length / extended_length  # interpolation scaler of RoPE
        self.weight = self._init_weight(self.weight)  # self.weight is from parent class "nn.Embedding"

    def _init_weight(self, out: nn.Parameter) -> nn.Parameter:
        """ positional interpolation version of RoPE
        self.scaler is same as expression in main idea (mL/L')

        Args:
            out (nn.Parameter): weight of nn.Embedding, which is parent class of current module

        return:
            re-define weight matrix for encoding the rotary position embedding, re-scaled by scaler,
            calculated by ratio (pretrained length / extended_length)
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[(pos * self.scaler) / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # half of hidden state
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0) -> Tensor:
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)  # super() is nn.Embedding(), super will return the embedding weight
