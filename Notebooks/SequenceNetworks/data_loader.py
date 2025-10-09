"""Data loader and associate utils functions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
from einops import pack, rearrange, repeat
from numpy.lib.npyio import NpzFile
from sklearn.preprocessing import StandardScaler
from torch.nested._internal.nested_tensor import NestedTensor
from torch.utils.data import Dataset


@dataclass
class SITS:
    """Data class for SITS with (nested) tensor.
    From: https://src.koda.cnrs.fr/mmdc/mtan_s1s2_classif/-/blob/master/src/mtan_s1s2_classif/models/torch/time_series.py
    """

    data: torch.Tensor | NestedTensor
    doy: torch.Tensor | NestedTensor
    mask: torch.Tensor | None = None

    def __post_init__(self):
        assert self.data.shape[0] == self.doy.shape[0]
        if self.mask is not None:
            assert isinstance(self.data, torch.Tensor)
            assert isinstance(self.doy, torch.Tensor)
            assert self.data.shape[0] == self.mask.shape[0]
        else:
            assert isinstance(self.data, NestedTensor)
            assert isinstance(self.doy, NestedTensor)

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    def to(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> Self:
        """
        Move sits to another device / dtype
        """
        self.data = self.data.to(device=device, dtype=dtype)

        if self.mask is not None:
            # We do not allow to cast mask dtype,
            # since it should always be of type torch.bool
            self.mask = self.mask.to(device=device)

        self.doy = self.doy.to(device=device, dtype=dtype)

        return self


def constant_pad_end(
    data: torch.Tensor, dim: int, length: int, fill_value: float | int | bool
) -> torch.Tensor | None:
    """
    Pad given dim at the end
    If data dim is larger than length, returns None
    """

    if data.shape[dim] < length:
        padding_length = length - data.shape[dim]
        padding_shape = list(data.shape)
        padding_shape[dim] = padding_length
        padd_data = torch.full(
            padding_shape, fill_value, dtype=data.dtype, device=data.device
        )
        return torch.cat((data, padd_data), dim=dim)
    if data.shape[dim] == length:
        return data
    return None


def pad_acquisition_time(
    sits: SITS, doy_length: int, fill_value: float = 0.0
) -> SITS | None:
    """
    Increase the size of the time_dimension to doy_length for
    all tensors in the SITS

    If the length of the time dimension is greater than doy_length, returns None
    """
    padded_data = constant_pad_end(sits.data, 1, doy_length, fill_value)
    padded_doy = constant_pad_end(sits.doy, 1, doy_length, fill_value)
    if padded_data is None or padded_doy is None:
        return None

    padded_mask: torch.Tensor | None = None
    if sits.mask is not None:
        padded_mask = constant_pad_end(
            sits.mask, 1, doy_length, False
        )  # True -> False !
        if padded_mask is None:
            return None

    return SITS(padded_data, padded_doy, padded_mask)


# Create data loader
class SITSDataSet(Dataset):
    """Dataset for SITS.

    Mask convention: 1 is valid and 0 is non-valid.
    """

    def __init__(
        self,
        data_path: str,
        mask_type: str = "mask",
        dtype: torch.dtype = torch.float32,
        scale: float = 10000.0,
        init_scaler: bool = False,
    ):

        # Check if repository exist
        self.data_path: str = data_path
        assert Path(self.data_path).exists()

        self.mask_type = mask_type
        self.dtype = dtype
        self.scale = scale
        self.scaler = None
        self.mean_: torch.Tensor | None = None
        self.std_: torch.Tensor | None = None

        # Get all files
        files = Path(self.data_path).glob("*.npz")
        assert files
        self.items: dict = dict(enumerate(files))
        if init_scaler:
            self.init_scaler()

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.items)

    def __getitem__(self, idx: int) -> SITS:
        """Read one item of the dataset."""
        with np.load(self.items[idx]) as item:
            data: torch.Tensor = (
                torch.from_numpy(
                    rearrange(item["data"], "t c h w-> (h w) t c"),
                ).to(self.dtype)
                / self.scale
            )
            if self.scaler:
                data = self.scaler(data)
            doy: torch.Tensor = torch.from_numpy(
                repeat(item["doy"], "t -> b t 1", b=data.shape[0])
            ).to(self.dtype)
            mask: torch.Tensor = torch.logical_not(
                torch.from_numpy(rearrange(self.get_mask(item), "t h w-> (h w) t"))
            )
        return SITS(data=data, doy=doy, mask=mask)

    def get_mask(self, item: NpzFile) -> np.ndarray:
        """Helper function returning mask/weight."""
        mask: np.ndarray
        match self.mask_type:
            case "mask":
                mask = np.any(item[self.mask_type], axis=1)
            case "weight":
                mask = item[self.mask_type]
        assert mask.ndim == 3
        return mask

    def remove_empty_patch(self) -> None:
        """Remove patchs all masked"""
        idx_remove = []
        for idx in range(len(self)):
            with np.load(self.items[idx]) as item:
                mask: torch.Tensor = torch.logical_not(
                    torch.from_numpy(rearrange(self.get_mask(item), "t h w-> (h w) t"))
                )
                if torch.any(mask.sum(dim=1) == 0):
                    idx_remove.append(idx)
        for idx_ in idx_remove:
            print(f"Remove file: {idx_}")
            self.items.pop(idx)

    def init_scaler(self):
        """Scaler for the data"""
        scaler = StandardScaler()
        for idx in range(len(self)):
            with np.load(self.items[idx]) as item:
                data = rearrange(item["data"], "t c h w-> (t h w) c") / self.scale

                mask = np.logical_not(rearrange(self.get_mask(item), "t h w-> (t h w)"))
                scaler.partial_fit(data[mask, :])

        self.mean_ = rearrange(
            torch.from_numpy(scaler.mean_).to(self.dtype), "c -> 1 1 c"
        )
        self.scale_ = rearrange(
            torch.from_numpy(scaler.scale_).to(self.dtype), "c -> 1 1 c"
        )
        self.scaler = lambda x: (x - self.mean_) / self.scale_
        self.inv_scaler = lambda x: x * self.scale_ + self.mean_


def collate_fn(list_of_dense_sits: list[SITS]) -> SITS:
    """Collate function to cat a list of dense SITS along the batch dimension."""
    # Get maximum size of time sample
    max_n_time = max(sits.shape[1] for sits in list_of_dense_sits)

    # Pad all sits to have the same size
    list_of_dense_sits_padded = [
        pad_acquisition_time(sits, doy_length=max_n_time) for sits in list_of_dense_sits
    ]

    # Cat everything
    return SITS(
        data=pack(
            [sits.data for sits in list_of_dense_sits_padded if sits is not None],
            "* t c",
        )[0],
        doy=pack(
            [sits.doy for sits in list_of_dense_sits_padded if sits is not None],
            "* t c",
        )[0],
        mask=pack(
            [sits.mask for sits in list_of_dense_sits_padded if sits is not None], "* t"
        )[0],
    )


def split_sits(sits: SITS, masked_size: float = 0.25) -> tuple[SITS, SITS]:
    """Split SITS w.r.t. time. Currently, does not take into account missing data."""
    seed = 0
    random_generator = np.random.default_rng(seed)
    size_masked = np.floor(sits.shape[1] * masked_size, dtype=int, casting="unsafe")

    # Suffle in time dimension
    idx = torch.from_numpy(
        random_generator.permutation(
            repeat(np.arange(sits.shape[1]), "t -> b t", b=sits.shape[0]), axis=1
        )
    ).to(sits.data.device)

    assert sits.mask is not None
    mask = sits.mask.gather(dim=1, index=idx)
    while torch.any(mask.sum(dim=1) == 0):  # Check empty sits
        print("Empty time: shuffle again")
        idx = torch.from_numpy(
            random_generator.permutation(
                np.arange(sits.shape[0], sits.shape[1]), axis=1
            )
        ).to(sits.data.device)
        mask = sits.mask.gather(dim=1, index=idx)
    data = sits.data.gather(dim=1, index=repeat(idx, "b t -> b t c", c=sits.shape[2]))
    doy = sits.doy.gather(dim=1, index=idx[..., None])

    return (
        SITS(
            data[:, size_masked:, :],
            doy=doy[:, size_masked:, :],
            mask=mask[:, size_masked:],
        ),
        SITS(
            data[:, :size_masked, :],
            doy=doy[:, :size_masked, :],
            mask=mask[:, :size_masked],
        ),
    )


def narrow_tensor(tensor: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    """
    Remove masked values from a 2D or 3D tensor, returning a list of per-sample tensors
    with varying lengths.

    Args:
        tensor: shape (B, T) or (B, T, C)
        mask:   shape (B, T), boolean mask

    Returns:
        list of tensors:
          - for 2D input: each element has shape (Tb,)
          - for 3D input: each element has shape (Tb, C)
          where Tb = number of True in mask[b] along dim T

    Array formulation of:
    def narrow_tensor(tensor, mask) -> list[torch.Tensor]:
        n_samples = tensor.shape[0]
        idx = mask == True
        if tensor.ndim == 2:
            return [data[i, idx[i, :]] for i in range(n_samples)]
        return [data[i, idx[i, :], :] for i in range(n_samples)]
    """
    assert mask.ndim == 2
    assert tensor.ndim in (2, 3)

    if tensor.ndim == 2:
        # Select all valid entries at once, flatten into 1D
        values = tensor[mask]
    else:
        # Expand mask to match last dim
        values = tensor[mask.unsqueeze(-1).expand_as(tensor)]
        # Reshape to (total_valid, C)
        values = rearrange(values, "(bt c) -> bt c", c=tensor.shape[-1])

    # Compute counts per row (number of True per b)
    lengths = mask.sum(dim=1).tolist()

    # Split back into per-row ragged tensors
    return list(torch.split(values, lengths))
