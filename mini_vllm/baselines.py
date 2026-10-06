"""Straightforward inference baselines for correctness and later benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from mini_vllm.cache import ContiguousKVCache
from mini_vllm.model import GPT2LM
from mini_vllm.scheduler import Request


def generate_naive(
    model: GPT2LM,
    prompt_tokens: Sequence[int] | Tensor,
    max_new_tokens: int,
    *,
    cache_capacity: int | None = None,
    on_token: Callable[[int, int, int], None] | None = None,
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
            if on_token is not None:
                on_token(0, step, token)
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


class _ContiguousCacheGroup:
    def __init__(self, caches: dict[int, ContiguousKVCache]) -> None:
        self.caches = caches

    def write(self, layer_idx, seq_id, positions, key, value):
        self.caches[seq_id].write(layer_idx, seq_id, positions, key, value)

    def read(self, layer_idx, seq_id, end_position):
        return self.caches[seq_id].read(layer_idx, seq_id, end_position)


def generate_static_batch(
    model: GPT2LM,
    requests: Sequence[Request],
    *,
    on_token: Callable[[int, int, int], None] | None = None,
) -> dict[int, list[int]]:
    """Run a fixed cohort together until its longest output is complete."""
    if not requests:
        return {}
    if any(request.max_new_tokens <= 0 or not request.prompt_tokens for request in requests):
        raise ValueError("static batches require non-empty prompts and positive output lengths")
    max_steps = max(request.max_new_tokens for request in requests)
    device = model.wte.weight.device
    cache_map = {
        request.request_id: ContiguousKVCache(
            model.config.num_layers,
            len(request.prompt_tokens) + request.max_new_tokens,
            model.config.num_heads,
            model.config.head_dim,
            device=device,
            dtype=model.wte.weight.dtype,
        )
        for request in requests
    }
    cache = _ContiguousCacheGroup(cache_map)
    max_prompt = max(len(request.prompt_tokens) for request in requests)
    input_ids = torch.zeros((len(requests), max_prompt), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, request in enumerate(requests):
        prompt = torch.tensor(request.prompt_tokens, dtype=torch.long, device=device)
        input_ids[row, : prompt.numel()] = prompt
        attention_mask[row, : prompt.numel()] = 1
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    outputs = {request.request_id: [] for request in requests}
    with torch.inference_mode():
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            seq_ids=[request.request_id for request in requests],
        )
        next_logits = torch.stack([logits[row, len(req.prompt_tokens) - 1] for row, req in enumerate(requests)])
        previous_tokens = []
        for request in requests:
            token = int(next_logits[len(previous_tokens)].argmax().item())
            outputs[request.request_id].append(token)
            if on_token is not None:
                on_token(request.request_id, 0, token)
            previous_tokens.append(token)

        prompt_lengths = torch.tensor([len(request.prompt_tokens) for request in requests], device=device)
        for output_index in range(1, max_steps):
            active = torch.tensor(
                [output_index < request.max_new_tokens for request in requests], dtype=torch.long, device=device
            )
            positions = (prompt_lengths + output_index - 1)[:, None]
            token_input = torch.tensor(previous_tokens, dtype=torch.long, device=device)[:, None]
            logits = model(
                token_input,
                attention_mask=active[:, None],
                position_ids=positions,
                cache=cache,
                seq_ids=[request.request_id for request in requests],
            )[:, -1]
            previous_tokens = []
            for row, request in enumerate(requests):
                if output_index < request.max_new_tokens:
                    token = int(logits[row].argmax().item())
                    outputs[request.request_id].append(token)
                    if on_token is not None:
                        on_token(request.request_id, output_index, token)
                    previous_tokens.append(token)
                else:
                    previous_tokens.append(outputs[request.request_id][-1])
    return outputs
