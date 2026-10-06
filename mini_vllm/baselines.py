"""Straightforward inference baselines for correctness and later benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from mini_vllm.cache import ContiguousKVCache
from mini_vllm.model import GPT2LM


def generate_naive(
    model: GPT2LM,
    prompt_tokens: Sequence[int] | Tensor,
    max_new_tokens: int,
    *,
    cache_capacity: int | None = None,
) -> list[int]:
    """Greedy-decode one request with its own contiguous KV cache."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    prompt = torch.as_tensor(prompt_tokens, dtype=torch.long, device=model.wte.weight.device).reshape(-1)
    if prompt.numel() == 0:
        raise ValueError("prompt must contain at least one token")
    total_capacity = prompt.numel() + max_new_tokens
    if total_capacity > model.config.max_position_embeddings:
        raise ValueError("prompt plus output exceeds model position embeddings")
    cache = ContiguousKVCache(
        model.config.num_layers,
        cache_capacity or total_capacity,
        model.config.num_heads,
        model.config.head_dim,
        device=prompt.device,
        dtype=model.wte.weight.dtype,
    )
    seq_id = 0
    with torch.inference_mode():
        logits = model(
            prompt[None, :],
            position_ids=torch.arange(prompt.numel(), device=prompt.device)[None, :],
            cache=cache,
            seq_ids=[seq_id],
        )[0, -1]
        generated: list[int] = []
        for step in range(max_new_tokens):
            token = int(logits.argmax().item())
            generated.append(token)
            if step + 1 < max_new_tokens:
                position = prompt.numel() + step
                next_input = torch.tensor([[token]], device=prompt.device)
                logits = model(
                    next_input,
                    position_ids=torch.tensor([[position]], device=prompt.device),
                    cache=cache,
                    seq_ids=[seq_id],
                )[0, -1]
    return generated
