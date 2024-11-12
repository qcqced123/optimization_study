import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, Optional
from einops.layers.torch import Rearrange


class LongRoPEScaledRotaryEmbedding(nn.Embedding):
    """
    reference:
        https://github.com/microsoft/LongRoPE/blob/main/rope/longrope.py
    """
    pass


class DynamicLongRoPEScaledRotaryEmbedding(LongRoPEScaledRotaryEmbedding):
    pass


def apply_rotary_position_embeddings(
    sinusoidal_pos: Tensor,
    query_layer: Tensor,
    key_layer: Tensor,
    value_layer: Tensor = None
):
    """ apply rotary position encoding to query, key layer
    Original Source code from Huggingface's RoFormer model, which is the most optimized way to create positional embedding

    You can find mathematical proof in official paper's Appendix

    Args:
        sinusoidal_pos: sinusoidal positional encoding, shape [batch(None), num_dim(None), seq_len, dim_head]
        query_layer: query matrix, shape (batch_size, num_head, seq_len, dim_head)
        key_layer: key matrix, shape (batch_size, num_head, seq_len, dim_head)
        value_layer: value matrix, shape (batch_size, num_head, seq_len, dim_head)

    References:
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py#L323
    """
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)  # select two element of index values
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

    # mathematical expression from Appendix in official repo
    # matmul the query matrix and rope matrix
    rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
        query_layer
    )
    query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos

    # matmul the key matrix and rope matrix
    rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
    key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos

    # in official, they don't use value_layer
    if value_layer is not None:
        rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
            value_layer
        )
        value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
        return query_layer, key_layer, value_layer

    return query_layer, key_layer


class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """ This module produces sinusoidal positional embeddings of any length
    Original Source code from Huggingface's RoFormer model, which is the most optimized way to create positional embedding

    Args:
        max_seq: max sequence length of model
        dim_head: dimension of each attention head's hidden states

    References:
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py#L323
    """

    def __init__(self, max_seq: int, dim_head: int) -> None:
        super().__init__(max_seq, dim_head)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """ identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
        The cos features are in the 2nd half of the vector. [dim // 2:]

        Args:
            out (nn.Parameter): weight matrix from parent class module 'nn.Embedding'

        return:
            re-define weight matrix for encoding the rotary position embedding
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
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


class Embedding(nn.Module):
    """ Class module for Roformer Embedding, word embedding & rotary positional encoding
    This module has option => whether or not to use ALBERT Style Factorized Embedding

    Very Un-Optimized way to apply rotary position encoding to word embedding
    Notes:
         ASAP, we will implement more optimized way to apply rotary position encoding to word embedding

    This Module set & initialize 3 Embedding Layers:
        1) Word Embedding
        2) Rotary Positional Encoding

    Args:
        cfg: configuration.py

    References:
        https://arxiv.org/abs/1706.03762
        https://arxiv.org/pdf/1810.04805.pdf
        https://arxiv.org/abs/2006.16236
        https://arxiv.org/abs/2104.09864  # RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
        https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
    """
    def __init__(self, cfg) -> None:
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.max_seq = cfg.max_seq
        self.dim_model = cfg.dim_model
        self.word_embedding = nn.Embedding(len(cfg.tokenizer), cfg.dim_model)
        self.layer_norm1 = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)  # for word embedding
        self.hidden_dropout = nn.Dropout(p=cfg.hidden_dropout_prob)
        # self.rotary_pos_encoding = self.create_rotation_matrix
        self.rotary_pos_encoding = RoFormerSinusoidalPositionalEmbedding(
            cfg.max_seq,
            cfg.dim_model // cfg.num_attention_heads
        )

        # ALBERT Style Factorized Embedding
        if self.cfg.is_mf_embedding:
            self.word_embedding = nn.Embedding(len(cfg.tokenizer), int(cfg.dim_model/6))
            self.projector = nn.Linear(int(cfg.dim_model/6), cfg.dim_model)  # project to original hidden dim

    @torch.no_grad()
    def create_rotation_matrix(self, seq_len: int) -> Tensor:
        """ Create a batch of rotation matrices from the given thetas.
        This function must be wrapped with torch.no_grad(), because it's not learnable parameters

        1) Create m*theta matrix (seq_len, dim_model): thetas
            - m: position index
            - theta: positional encoding value from static function (10000**(-2 * (i_arr - 1) / self.dim_model))

        2) Create R matrix (seq_len, dim_model, dim_model): R
            - example:
                [cos m*theta1, -sin m*theta1, 0, 0]
                [sin m*theta1, cos m*theta1, 0, 0]
                [0, 0, cos m*theta2, -sin m*theta2]
                [0, 0, sin m*theta2, cos m*theta2]

        Args:
            seq_len: max sequence length in batch

        Returns:
            Tensor: A tensor of shape (batch_size, seq_len, d, d) containing the rotation matrices.
        """
        i_arr = torch.arange(1, int(self.dim_model / 2) + 1).repeat_interleave(2).to(self.cfg.device)  # for rotary position embedding
        theta = 10000**(-2 * (i_arr - 1) / self.dim_model)  # for rotary position embedding
        scaler = torch.arange(1, seq_len + 1, device=self.cfg.device, dtype=torch.float).unsqueeze(1).repeat(1, self.dim_model).reshape(seq_len, self.dim_model)
        thetas = torch.mul(scaler, theta)

        R = torch.eye(self.dim_model, device=thetas.device).repeat(seq_len, 1, 1)
        for i in range(0, self.dim_model, 2):
            cos_t = torch.cos(thetas[:, i]).unsqueeze(-1)
            sin_t = torch.sin(thetas[:, i]).unsqueeze(-1)

            R[:, i, i] = cos_t.squeeze(-1)
            R[:, i + 1, i + 1] = cos_t.squeeze(-1)
            R[:, i, i + 1] = -sin_t.squeeze(-1)
            R[:, i + 1, i] = sin_t.squeeze(-1)

        return R

    def forward(self, inputs: Tensor) -> Tuple[nn.Embedding, Tensor]:
        if self.cfg.is_mf_embedding:
            word_embeddings = self.hidden_dropout(
                self.layer_norm1(self.projector(self.word_embedding(inputs)))
            )
        else:
            word_embeddings = self.hidden_dropout(
                self.layer_norm1(self.word_embedding(inputs))
            )
        rotary_pos_enc = self.rotary_pos_encoding(inputs.shape[1])
        return word_embeddings, rotary_pos_enc
