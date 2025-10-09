"""Implement transformer class for nested tensor."""

import torch
from torch import nn

from transformer_blocks import (
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class FourrierPositionalEncoding(nn.Module):
    def __init__(
        self,
        div: float = 1000,
        offset: int = 0,
        nb_features: int = 64,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.div = div
        self.offset = offset
        self.nb_features = nb_features
        self.denom_layer = nn.Linear(
            1, self.nb_features // 2, bias=False, device=device, dtype=dtype
        )
        with torch.no_grad():  # trick to use nested tensors later, slicing is not possible with them
            self.denom_layer.weight = nn.Parameter(
                torch.pow(
                    1 / self.div,
                    2
                    * (
                        torch.arange(
                            self.offset,
                            self.offset + nb_features // 2,
                            device=device,
                            dtype=dtype,
                        ).float()
                    )
                    / nb_features,
                ).unsqueeze(1),
                requires_grad=False,
            )

    def forward(
        self,
        doy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward method
        """
        pe = self.denom_layer(doy)
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)
        return pe


class Transformer(nn.Module):
    """Transfomer usable with nested tensor."""

    def __init__(
        self,
        n_channels: int,
        n_heads: int = 2,
        emb_dim: int = 64,
        depth: int = 1,
        dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # init
        factory_kwargs = {"device": device, "dtype": dtype}
        self.n_channels = n_channels
        self.n_head = n_heads
        self.emb_dim = emb_dim
        self.depth = depth

        self.time_embedding = FourrierPositionalEncoding(
            nb_features=emb_dim, **factory_kwargs
        )
        self.linear_encoding = nn.Linear(
            self.n_channels, self.emb_dim, **factory_kwargs
        )
        self.linear_decoding = nn.Linear(
            self.emb_dim, self.n_channels, **factory_kwargs
        )

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for _ in range(depth):
            self.encoder_blocks.append(
                TransformerEncoderBlock(
                    emb_dim=emb_dim, n_heads=n_heads, dropout=dropout, **factory_kwargs
                )
            )
            self.decoder_blocks.append(
                TransformerDecoderBlock(
                    emb_dim=emb_dim, n_heads=n_heads, dropout=dropout, **factory_kwargs
                )
            )

    def forward(
        self, value: torch.Tensor, time_key: torch.Tensor, time_query: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the tansformer."""

        # Do embedding and add positional encoding
        value_embedded = self.linear_encoding(value)
        time_key_embedded = torch.nested.nested_tensor_from_jagged(  # workaround to sum to nested tensor of same shape
            values=self.time_embedding(time_key).values(),
            offsets=value_embedded.offsets(),  # https://docs.pytorch.org/docs/stable/nested.html#data-layout-and-shape
        )
        latent_value = value_embedded + time_key_embedded

        # Do transformer encoding
        for d in range(self.depth):
            latent_value = self.encoder_blocks[d](latent_value)

        # Do transformer decoding
        # here we just need to encode time. Since we don't have corresponding value
        # and since we are doing "add" positional encoding, it is equivalent
        # to set missing value to zeros
        time_query_embedded = self.time_embedding(time_query)
        output = self.decoder_blocks[0](latent_value, time_query_embedded)
        for d in range(1, self.depth):
            output = self.decoder_blocks[d](latent_value, output)

        output = self.linear_decoding(output)
        return output
