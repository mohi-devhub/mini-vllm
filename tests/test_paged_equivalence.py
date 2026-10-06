import torch

from mini_vllm.cache import BlockPool, ContiguousKVCache, PagedKVCache
from mini_vllm.model import GPT2Config, GPT2LM


class CacheGroup:
    """Route the model's cache calls to one contiguous allocation per sequence."""

    def __init__(self, caches: dict[int, ContiguousKVCache]) -> None:
        self.caches = caches

    def write(self, layer_idx, seq_id, positions, key, value):
        self.caches[seq_id].write(layer_idx, seq_id, positions, key, value)

    def read(self, layer_idx, seq_id, end_position):
        return self.caches[seq_id].read(layer_idx, seq_id, end_position)


def test_paged_and_contiguous_logits_match_for_mixed_lengths_and_decode() -> None:
    torch.manual_seed(23)
    model = GPT2LM(
        GPT2Config(vocab_size=83, max_position_embeddings=64, hidden_size=48, num_layers=2, num_heads=4)
    ).eval()
    block_size = 4
    pool = BlockPool(2, 12, block_size, 4, 12, device="cpu", dtype=torch.float32)
    paged = PagedKVCache(pool)
    seq_ids = [101, 202]
    paged.add_sequence(seq_ids[0])
    paged.add_sequence(seq_ids[1])
    contiguous = CacheGroup(
        {
            seq_id: ContiguousKVCache(2, 32, 4, 12, device="cpu", dtype=torch.float32)
            for seq_id in seq_ids
        }
    )

    input_ids = torch.tensor([[5, 11, 17, 23, 29, 0, 0, 0, 0], [7, 13, 19, 25, 31, 37, 43, 49, 55]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    with torch.inference_mode():
        paged_prefill = model(input_ids, attention_mask, position_ids, paged, seq_ids)
        contiguous_prefill = model(input_ids, attention_mask, position_ids, contiguous, seq_ids)
        valid_logits = attention_mask.bool()
        torch.testing.assert_close(
            paged_prefill[valid_logits], contiguous_prefill[valid_logits], rtol=1e-5, atol=1e-5
        )

        lengths = attention_mask.sum(-1)
        decode_positions = lengths[:, None]
        decode_tokens = torch.tensor([[61], [67]])
        paged_decode = model(
            decode_tokens,
            position_ids=decode_positions,
            cache=paged,
            seq_ids=seq_ids,
        )
        contiguous_decode = model(
            decode_tokens,
            position_ids=decode_positions,
            cache=contiguous,
            seq_ids=seq_ids,
        )
    torch.testing.assert_close(paged_decode, contiguous_decode, rtol=1e-5, atol=1e-5)
    assert pool.blocks_in_use == 5  # 2 blocks for the 5-token prompt, 3 for the 9-token prompt.
