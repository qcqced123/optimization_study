"""py module for implementing the YaRN from EleutherAI Research
"""
import torch
import numpy as np
import torch.nn as nn

from math import log, pi
from torch import Tensor


class YaRNScaledRotaryEmbedding(nn.Module):
    """ module of YaRN (Yet another RoPE Extend method)

    main idea:
        ntk-by-parts with rescaling self-attention softmax temperature
        [ntk-by-parts]
        - f′(x, m, θ) = f(x, g(m), h(θ))
        - f: original RoPE Function
        - g(m): map between real numbers
        - h(θ): acts on the entries of the diagonal matrix θ, uniformly by diag(h(θ1), · · · , h(θ|D|/2))

        [rescaling softmax temperature]

    Args:
        extended_length (int): target value of extended context window size
        pretrained_length (int): value of pretrained model's context window size
        dim_head (int): value of size of pretrained model's hidden state

    Reference:
        https://arxiv.org/abs/2309.00071  # original paper
        https://simpling.tistory.com/56  # well explain about neural tangent kernel
        https://pytorch.org/tutorials/intermediate/neural_tangent_kernels.html  # NTK in pytorch
        https://github.com/microsoft/LongRoPE/blob/main/rope/yarn.py
    """
    def __init__(self, extended_length: int, pretrained_length: int, dim_head: int, alpha: int = 1, beta: int = 32) -> None:
        super().__init__(extended_length, dim_head)
        self.dim_head = dim_head
        self.pretrained_length = pretrained_length
        self.scaler = pretrained_length / extended_length  # interpolation scaler of RoPE
        self.temperature = 0.1 * log(self.scaler) + 1
        self.alpha = alpha  # upper bound of wavelength of low frequency from official paper
        self.beta = beta  # lower bound of wavelength of high frequency from official paper
        self.weight = self._init_weight(self.weight)  # self.weight is from parent class "nn.Embedding"

    def ramp_func(self, d: int):
        """ class method for calculating the wavelength of d-th hidden state, ramp value
        ramp value is used to determine the interpolation method of dimension of hidden state

        Args:
            d (int): number of current hidden state
x
        Return: ramp value, γ(r) (in official paper expression)
        """
        r_d = self.pretrained_length / np.power((2 * pi * 10000), abs(2 * d / self.dim_head))
        if r_d < self.alpha:
            ramp = 0
        elif r_d > self.beta:
            ramp = 1
        else:
            ramp = (r_d - self.alpha) / (self.beta - self.alpha)
        return ramp

    def _init_weight(self, out: nn.Parameter) -> Tensor:
        # make the position_encoding array of YaRN
        n_pos, dim = out.shape
        position_enc = []
        for pos in range(n_pos):
            cnt = []
            for j in range(dim):
                theta_d = np.power(10000, 2 * (j // 2) / dim)
                ramp = self.ramp_func(j)
                pos_value = pos * ((1 - ramp) * (theta_d / self.scaler) + (ramp * theta_d))
                cnt.append(pos_value)
            position_enc.append(cnt)

        position_enc = np.array(position_enc)
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # half of hidden state
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out * self.temperature  # ASAP, check this logic

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0) -> Tensor:
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)  # super() is nn.Embedding(), super will return the embedding weight
