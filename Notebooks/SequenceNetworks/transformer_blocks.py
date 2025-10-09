""" Transformer blocks with nested tensor"""

import torch
import torch.nn.functional as F
from torch import nn


# TODO: split query key value in encoder
class MultiHeadAttention(nn.Module):
    """

    From: https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        emb_q (int): Size of embedding dim for query
        emb_k (int): Size of embedding dim for key
        emb_v (int): Size of embedding dim for value
        emb_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim emb_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        emb_q: int,
        emb_k: int,
        emb_v: int,
        emb_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert emb_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = emb_q == emb_k == emb_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(
                emb_q, emb_total * 3, bias=bias, **factory_kwargs
            )
        else:
            self.q_proj = nn.Linear(emb_q, emb_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(emb_k, emb_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(emb_v, emb_total, bias=bias, **factory_kwargs)
        emb_out = emb_q
        self.out_proj = nn.Linear(emb_total, emb_out, bias=bias, **factory_kwargs)
        self.emb_head = emb_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape ('N', 'L_q', 'emb_qk')
            key (torch.Tensor): key of shape ('N', 'L_kv', 'emb_qk')
            value (torch.Tensor): value of shape ('N', 'L_kv', 'emb_v')

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, emb_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, emb_total) -> (N, L_t, nheads, emb_head) -> (N, nheads, L_t, emb_head)
        query = query.unflatten(-1, [self.nheads, self.emb_head]).transpose(1, 2)
        # (N, L_s, emb_total) -> (N, L_s, nheads, emb_head) -> (N, nheads, L_s, emb_head)
        key = key.unflatten(-1, [self.nheads, self.emb_head]).transpose(1, 2)
        # (N, L_s, emb_total) -> (N, L_s, nheads, emb_head) -> (N, nheads, L_s, emb_head)
        value = value.unflatten(-1, [self.nheads, self.emb_head]).transpose(1, 2)

        # # Step 3. Run SDPA
        # # (N, nheads, L_t, emb_head)
        # attn_output = F.scaled_dot_product_attention(
        #     query, key, value, dropout_p=self.dropout
        # )
        # # (N, nheads, L_t, emb_head) -> (N, L_t, nheads, emb_head) -> (N, L_t, emb_total)
        # attn_output = attn_output.transpose(1, 2).flatten(-2)

        # # Step 4. Apply output projection
        # # (N, L_t, emb_total) -> (N, L_t, emb_out)
        # attn_output = self.out_proj(attn_output)

        attn_output = self.out_proj(
            F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout)
            .transpose(1, 2)
            .flatten(-2)
        )

        return attn_output


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int = 64,
        n_heads: int = 2,
        feed_forward_hidden_dim: int = 512,
        dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.multi_head_attention = MultiHeadAttention(
            emb_dim,
            emb_dim,
            emb_dim,
            emb_dim,
            nheads=n_heads,
            dropout=0.0,
            **factory_kwargs,
        )
        # layer normalisation
        self.layer_norm_1 = nn.LayerNorm(emb_dim, **factory_kwargs)
        self.layer_norm_2 = nn.LayerNorm(emb_dim, **factory_kwargs)

        # Feed Forward
        self.feed_forward_layers = nn.Sequential(
            nn.Linear(
                emb_dim,
                feed_forward_hidden_dim,
                **factory_kwargs,
            ),
            nn.ReLU(),
            nn.Linear(
                feed_forward_hidden_dim,
                emb_dim,
                **factory_kwargs,
            ),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_key_value: torch.Tensor):
        z = self.dropout(
            self.layer_norm_1(
                self.multi_head_attention(
                    query_key_value, query_key_value, query_key_value
                )
                + query_key_value
            )
        )
        return self.dropout(self.layer_norm_2(self.feed_forward_layers(z) + z))


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int = 64,
        n_heads: int = 2,
        feed_forward_hidden_dim: int = 512,
        dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.multi_head_self_attention = MultiHeadAttention(
            emb_dim,
            emb_dim,
            emb_dim,
            emb_dim,
            nheads=n_heads,
            dropout=0.0,
            **factory_kwargs,
        )
        self.multi_head_cross_attention = MultiHeadAttention(
            emb_dim,
            emb_dim,
            emb_dim,
            emb_dim,
            nheads=n_heads,
            dropout=0.0,
            **factory_kwargs,
        )

        # layer normalisation
        self.layer_norm_1 = nn.LayerNorm(emb_dim, **factory_kwargs)
        self.layer_norm_2 = nn.LayerNorm(emb_dim, **factory_kwargs)
        self.layer_norm_3 = nn.LayerNorm(emb_dim, **factory_kwargs)

        # Feed Forward
        self.feed_forward_layers = nn.Sequential(
            nn.Linear(emb_dim, feed_forward_hidden_dim, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden_dim, emb_dim, **factory_kwargs),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, query: torch.Tensor):
        # Self attention
        z = self.dropout(
            self.layer_norm_1(
                self.multi_head_self_attention(query, query, query) + query
            )
        )

        # Cross attention
        z = self.dropout(
            self.layer_norm_2(self.multi_head_cross_attention(z, x, x) + z)
        )

        return self.dropout(self.layer_norm_3(self.feed_forward_layers(z) + z))
