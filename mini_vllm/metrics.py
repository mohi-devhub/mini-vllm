"""Timing helpers and summary statistics shared by the benchmark runner."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

from mini_vllm.engine import Engine
from mini_vllm.scheduler import Request


def synchronize(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def timed_trials(
    run_once: Callable[[], Any],
    *,
    device: torch.device | str,
    trials: int = 3,
    warmup: Callable[[], Any] | None = None,
) -> list[dict[str, Any]]:
    if trials < 3:
        raise ValueError("benchmark measurements require at least three trials")
    if warmup is not None:
        with torch.inference_mode():
            warmup()
        synchronize(device)
    results = []
    for _ in range(trials):
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            payload = run_once()
        synchronize(device)
        elapsed = time.perf_counter() - start
        peak_memory = (
            torch.cuda.max_memory_allocated(torch.device(device))
            if torch.device(device).type == "cuda"
            else 0
        )
        results.append({"elapsed_seconds": elapsed, "peak_gpu_memory_bytes": peak_memory, "payload": payload})
    return results


def summarize_engine_run(
    requests: Sequence[Request],
    elapsed_seconds: float,
    peak_gpu_memory_bytes: int,
    engine: Engine,
) -> dict[str, Any]:
    output_count = sum(len(request.generated_tokens) for request in requests)
    ttft = [request.first_token_time or 0.0 for request in requests]
    latency = [request.finish_time or 0.0 for request in requests]
    bytes_per_token = (
        engine.model.config.num_layers
        * 2
        * engine.model.config.num_heads
        * engine.model.config.head_dim
        * engine.model.wte.weight.element_size()
    )
    result = {
        "throughput_output_tokens_per_second": output_count / elapsed_seconds if elapsed_seconds else 0.0,
        "time_to_first_token_seconds_p50": percentile(ttft, 50),
        "time_to_first_token_seconds_p95": percentile(ttft, 95),
        "latency_seconds_p50": percentile(latency, 50),
        "latency_seconds_p95": percentile(latency, 95),
        "latency_seconds_p99": percentile(latency, 99),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "max_concurrent_sequences": engine.max_concurrent_sequences,
        "preemptions": engine.preemptions,
    }
    if engine.pool is not None:
        result["peak_kv_cache_bytes"] = engine.pool.peak_blocks * engine.pool.block_size * bytes_per_token
        result.update(
            {
                "peak_blocks_in_use": engine.pool.peak_blocks,
                "peak_block_utilization": engine.pool.peak_blocks / engine.pool.num_blocks,
                "peak_internal_fragmentation_tokens": engine.cache.peak_internal_fragmentation_tokens,
                "configured_block_count": engine.pool.num_blocks,
                "block_size": engine.pool.block_size,
            }
        )
    else:
        result["peak_kv_cache_bytes"] = engine.cache.peak_reserved_tokens * bytes_per_token
        result.update(
            {
                "peak_reserved_tokens": engine.cache.peak_reserved_tokens,
                "configured_token_capacity": engine.cache.capacity_tokens,
            }
        )
    return result
