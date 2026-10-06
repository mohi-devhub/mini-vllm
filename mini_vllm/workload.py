"""Deterministic synthetic request traces with heavy-tailed lengths."""

from __future__ import annotations

import random

from mini_vllm.scheduler import Request


def generate_trace(
    num_requests: int,
    arrival_rate: float,
    *,
    seed: int = 0,
    vocab_size: int = 50_257,
    min_prompt_tokens: int = 4,
    max_prompt_tokens: int = 256,
    min_output_tokens: int = 4,
    max_output_tokens: int = 128,
) -> list[Request]:
    """Create a Poisson arrival process and Pareto-tailed lengths."""
    if num_requests < 0 or arrival_rate <= 0:
        raise ValueError("num_requests must be non-negative and arrival_rate positive")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    rng = random.Random(seed)
    arrival_time = 0.0
    requests = []
    for request_id in range(num_requests):
        arrival_time += rng.expovariate(arrival_rate)
        prompt_length = min(
            max_prompt_tokens,
            min_prompt_tokens + int((rng.paretovariate(2.2) - 1.0) * 12),
        )
        output_length = min(
            max_output_tokens,
            min_output_tokens + int((rng.paretovariate(1.8) - 1.0) * 10),
        )
        prompt = [rng.randrange(1, vocab_size) for _ in range(prompt_length)]
        requests.append(Request(request_id, prompt, output_length, arrival_time))
    return requests
