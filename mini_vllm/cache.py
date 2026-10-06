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


class BlockPool:
    """Fixed K/V storage shared across layers by physical block ID."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if min(num_layers, num_blocks, block_size, num_heads, head_dim) <= 0:
            raise ValueError("block pool dimensions must be positive")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.storage = torch.empty(
            (num_layers, 2, num_blocks, block_size, num_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._free = list(range(num_blocks))
        self._allocated: set[int] = set()
        self.peak_blocks = 0

    @property
    def blocks_in_use(self) -> int:
        return len(self._allocated)

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise MemoryError("paged KV block pool exhausted")
        block_id = self._free.pop()
        self._allocated.add(block_id)
        self.peak_blocks = max(self.peak_blocks, len(self._allocated))
        return block_id

    def free(self, block_id: int) -> None:
        if block_id not in self._allocated:
            raise ValueError(f"block {block_id} is not allocated")
        self._allocated.remove(block_id)
        self._free.append(block_id)


class PagedKVCache:
    """Per-sequence block tables over a shared, preallocated pool."""

    def __init__(self, pool: BlockPool) -> None:
        self.pool = pool
        self.block_tables: dict[int, list[int]] = {}
        self._written_positions: dict[int, set[int]] = {}
        self.peak_internal_fragmentation_tokens = 0

    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self.block_tables:
            raise ValueError(f"sequence {seq_id} already has a block table")
        self.block_tables[seq_id] = []
        self._written_positions[seq_id] = set()

    def _ensure_sequence(self, seq_id: int) -> None:
        if seq_id not in self.block_tables:
            self.add_sequence(seq_id)

    def write(
        self,
        layer_idx: int,
        seq_id: int,
        positions: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> None:
        self._ensure_sequence(seq_id)
        positions_list = [int(position) for position in positions.tolist()]
        if len(positions_list) != key.shape[1] or key.shape != value.shape:
            raise ValueError("positions and K/V token dimensions must match")
        table = self.block_tables[seq_id]
        for position in positions_list:
            if position < 0:
                raise ValueError("cache positions cannot be negative")
            logical_block = position // self.pool.block_size
            while logical_block >= len(table):
                table.append(self.pool.allocate())
        for token_index, position in enumerate(positions_list):
            logical_block, offset = divmod(position, self.pool.block_size)
            physical_block = table[logical_block]
            self.pool.storage[layer_idx, 0, physical_block, offset].copy_(key[:, token_index, :])
            self.pool.storage[layer_idx, 1, physical_block, offset].copy_(value[:, token_index, :])
        self._written_positions[seq_id].update(positions_list)
        self.peak_internal_fragmentation_tokens = max(
            self.peak_internal_fragmentation_tokens,
            self.internal_fragmentation_tokens,
        )

    def read(self, layer_idx: int, seq_id: int, end_position: int) -> tuple[Tensor, Tensor]:
        table = self.block_tables[seq_id]
        if end_position and not set(range(end_position)).issubset(self._written_positions[seq_id]):
            raise ValueError(f"sequence {seq_id} has unwritten cache positions before {end_position}")
        # This gather emulates the block-table indirection a custom paged-attention CUDA kernel uses.
        key_parts = []
        value_parts = []
        remaining = end_position
        for physical_block in table:
            take = min(remaining, self.pool.block_size)
            if take <= 0:
                break
            key_parts.append(self.pool.storage[layer_idx, 0, physical_block, :take])
            value_parts.append(self.pool.storage[layer_idx, 1, physical_block, :take])
            remaining -= take
        if remaining:
            raise ValueError(f"sequence {seq_id} does not have {end_position} allocated positions")
        if not key_parts:
            shape = self.pool.storage.shape
            empty = self.pool.storage.new_empty((shape[-2], 0, shape[-1]))
            return empty, empty
        return (
            torch.cat(key_parts, dim=0).transpose(0, 1),
            torch.cat(value_parts, dim=0).transpose(0, 1),
        )

    def free_sequence(self, seq_id: int) -> None:
        table = self.block_tables.pop(seq_id)
        self._written_positions.pop(seq_id)
        for block_id in table:
            self.pool.free(block_id)

    @property
    def internal_fragmentation_tokens(self) -> int:
        allocated_slots = sum(len(table) * self.pool.block_size for table in self.block_tables.values())
        used_slots = sum(len(positions) for positions in self._written_positions.values())
        return allocated_slots - used_slots


class ContiguousCachePool:
    """Continuous-batch cache that reserves each request's full token capacity."""

    def __init__(
        self,
        num_layers: int,
        capacity_tokens: int,
        num_heads: int,
        head_dim: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.capacity_tokens = capacity_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.caches: dict[int, ContiguousKVCache] = {}
        self.reserved_tokens: dict[int, int] = {}
        self.peak_reserved_tokens = 0

    @property
    def free_tokens(self) -> int:
        return self.capacity_tokens - sum(self.reserved_tokens.values())

    @property
    def reserved_total(self) -> int:
        return sum(self.reserved_tokens.values())

    def add_sequence(self, seq_id: int, capacity: int) -> None:
        if seq_id in self.caches:
            raise ValueError(f"sequence {seq_id} already has a contiguous cache")
        if capacity > self.free_tokens:
            raise MemoryError("contiguous KV reservation exceeds the token capacity")
        self.caches[seq_id] = ContiguousKVCache(
            self.num_layers,
            capacity,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.reserved_tokens[seq_id] = capacity
        self.peak_reserved_tokens = max(self.peak_reserved_tokens, self.reserved_total)

    def write(self, layer_idx: int, seq_id: int, positions: Tensor, key: Tensor, value: Tensor) -> None:
        self.caches[seq_id].write(layer_idx, seq_id, positions, key, value)

    def read(self, layer_idx: int, seq_id: int, end_position: int) -> tuple[Tensor, Tensor]:
        return self.caches[seq_id].read(layer_idx, seq_id, end_position)

    def free_sequence(self, seq_id: int) -> None:
        self.caches.pop(seq_id)
        self.reserved_tokens.pop(seq_id)
