import pytest
import torch

from mini_vllm.cache import BlockPool, PagedKVCache


def _pool(blocks: int = 3, block_size: int = 4) -> BlockPool:
    return BlockPool(2, blocks, block_size, 2, 4, device="cpu", dtype=torch.float32)


def test_allocate_free_reuse_and_peak_count() -> None:
    pool = _pool()
    first = pool.allocate()
    second = pool.allocate()
    assert pool.blocks_in_use == 2
    pool.free(first)
    assert pool.allocate() == first
    assert pool.blocks_in_use == 2
    assert pool.peak_blocks == 2
    pool.free(second)
    pool.free(first)
    assert pool.blocks_in_use == 0
    assert pool.free_blocks == 3


def test_exhaustion_and_double_free_are_reported() -> None:
    pool = _pool(blocks=1)
    block = pool.allocate()
    with pytest.raises(MemoryError, match="exhausted"):
        pool.allocate()
    pool.free(block)
    with pytest.raises(ValueError, match="not allocated"):
        pool.free(block)


def test_sequence_table_releases_every_allocated_block() -> None:
    cache = PagedKVCache(_pool(blocks=4, block_size=4))
    cache.add_sequence(12)
    positions = torch.arange(9)
    values = torch.zeros((2, 9, 4))
    cache.write(0, 12, positions, values, values)
    assert len(cache.block_tables[12]) == 3
    assert cache.pool.blocks_in_use == 3
    assert cache.internal_fragmentation_tokens == 3
    cache.free_sequence(12)
    assert cache.pool.blocks_in_use == 0
    assert cache.pool.free_blocks == 4
