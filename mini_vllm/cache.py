"""KV cache implementations used by the model and the comparison baselines."""

from __future__ import annotations

import torch
from torch import Tensor


class ContiguousKVCache:
    """One sequence's contiguous cache, reserving capacity up front."""

    def __init__(
        self,
        num_layers: int,
        capacity: int,
        num_heads: int,
        head_dim: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.capacity = capacity
        self.keys = torch.empty((num_layers, capacity, num_heads, head_dim), device=device, dtype=dtype)
        self.values = torch.empty_like(self.keys)
        self.length = 0

    def write(
        self,
        layer_idx: int,
        seq_id: int,
        positions: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> None:
        del seq_id  # This cache instance belongs to exactly one sequence.
        positions = positions.to(device=self.keys.device, dtype=torch.long)
        if positions.numel() and int(positions.max().item()) >= self.capacity:
            raise MemoryError(f"cache capacity {self.capacity} exceeded")
        self.keys[layer_idx, positions] = key.transpose(0, 1)
        self.values[layer_idx, positions] = value.transpose(0, 1)
        if positions.numel():
            self.length = max(self.length, int(positions.max().item()) + 1)

    def read(self, layer_idx: int, seq_id: int, end_position: int) -> tuple[Tensor, Tensor]:
        del seq_id
        if end_position > self.length:
            raise ValueError(f"requested {end_position} cache positions, but only {self.length} are written")
        return (
            self.keys[layer_idx, :end_position].transpose(0, 1),
            self.values[layer_idx, :end_position].transpose(0, 1),
        )
