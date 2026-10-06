"""Run fair inference baselines over the same replayed synthetic traces."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_vllm.baselines import generate_naive, generate_static_batch
from mini_vllm.engine import Engine
from mini_vllm.metrics import summarize_engine_run
from mini_vllm.model import GPT2LM
from mini_vllm.scheduler import Request
from mini_vllm.workload import generate_trace


def _copy_requests(requests: list[Request], *, zero_arrivals: bool = False) -> list[Request]:
    clones = copy.deepcopy(requests)
    for request in clones:
        request.generated_tokens.clear()
        request.first_token_time = None
        request.finish_time = None
        request.prefill_logits = None
        if zero_arrivals:
            request.arrival_time = 0.0
    return clones


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_configuration(
    name: str,
    model: GPT2LM,
    trace: list[Request],
    *,
    device: torch.device,
    batch_size: int,
    num_blocks: int,
    block_size: int,
) -> dict:
    requests = _copy_requests(trace)
    started = time.perf_counter()
    if name == "naive_sequential":
        for request in sorted(requests, key=lambda item: item.arrival_time):
            time.sleep(max(0.0, request.arrival_time - (time.perf_counter() - started)))

            def on_token(_sequence: int, index: int, _token: int, req=request) -> None:
                elapsed = time.perf_counter() - started - req.arrival_time
                if index == 0:
                    req.first_token_time = elapsed
                if index + 1 == req.max_new_tokens:
                    req.finish_time = elapsed

            request.generated_tokens = generate_naive(
                model,
                request.prompt_tokens,
                request.max_new_tokens,
                on_token=on_token,
            )
        engine = None
    elif name == "static_batching":
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            arrival = max(request.arrival_time for request in batch)
            time.sleep(max(0.0, arrival - (time.perf_counter() - started)))

            def on_token(request_id: int, index: int, _token: int) -> None:
                req = next(request for request in batch if request.request_id == request_id)
                elapsed = time.perf_counter() - started - req.arrival_time
                if index == 0:
                    req.first_token_time = elapsed
                if index + 1 == req.max_new_tokens:
                    req.finish_time = elapsed

            outputs = generate_static_batch(model, batch, on_token=on_token)
            for request in batch:
                request.generated_tokens = outputs[request.request_id]
        engine = None
    else:
        engine = Engine(
            model,
            num_blocks=num_blocks,
            block_size=block_size,
            max_batch_size=batch_size,
            cache_mode="paged" if name == "paged_engine" else "contiguous",
        )
        engine.run(requests)
    _sync(device)
    elapsed = time.perf_counter() - started
    return {"requests": requests, "engine": engine, "elapsed_seconds": elapsed}


def _summarize(name: str, payload: dict, peak_gpu_memory: int, batch_size: int) -> dict:
    requests = payload["requests"]
    engine = payload["engine"]
    elapsed = payload["elapsed_seconds"]
    if engine is not None:
        return summarize_engine_run(requests, elapsed, peak_gpu_memory, engine)
    output_count = sum(len(request.generated_tokens) for request in requests)
    first_token = [request.first_token_time or 0.0 for request in requests]
    latency = [request.finish_time or 0.0 for request in requests]
    from mini_vllm.metrics import percentile

    return {
        "throughput_output_tokens_per_second": output_count / elapsed if elapsed else 0.0,
        "time_to_first_token_seconds_p50": percentile(first_token, 50),
        "time_to_first_token_seconds_p95": percentile(first_token, 95),
        "latency_seconds_p50": percentile(latency, 50),
        "latency_seconds_p95": percentile(latency, 95),
        "latency_seconds_p99": percentile(latency, 99),
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "max_concurrent_sequences": 1 if name == "naive_sequential" else min(batch_size, len(requests)),
        "preemptions": 0,
    }


def _plot(rows: list[dict], output_dir: Path) -> None:
    os.environ["MPLCONFIGDIR"] = str(output_dir / ".matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(output_dir / ".cache")
    (output_dir / ".matplotlib").mkdir(parents=True, exist_ok=True)
    (output_dir / ".cache").mkdir(parents=True, exist_ok=True)
    font_cache = output_dir / ".fontconfig-cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    font_config = output_dir / ".fontconfig.conf"
    font_config.write_text(
        "<?xml version=\"1.0\"?>\n"
        "<!DOCTYPE fontconfig SYSTEM \"urn:fontconfig:fonts.dtd\">\n"
        "<fontconfig>\n"
        "  <dir>/usr/share/fonts</dir>\n"
        "  <dir>/usr/local/share/fonts</dir>\n"
        "  <dir>/System/Library/Fonts</dir>\n"
        f"  <cachedir>{font_cache}</cachedir>\n"
        "</fontconfig>\n"
    )
    os.environ["FONTCONFIG_FILE"] = str(font_config)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = sorted({row["configuration"] for row in rows})
    rates = sorted({row["arrival_rate"] for row in rows})
    for metric, title, filename in (
        ("throughput_output_tokens_per_second", "Output throughput vs arrival rate", "throughput_vs_rate.png"),
        ("latency_seconds_p95", "P95 end-to-end latency vs arrival rate", "p95_latency_vs_rate.png"),
    ):
        fig, axis = plt.subplots(figsize=(8, 5))
        for config in configs:
            xs, ys = [], []
            for rate in rates:
                samples = [row[metric] for row in rows if row["configuration"] == config and row["arrival_rate"] == rate]
                if samples:
                    xs.append(rate)
                    ys.append(statistics.median(samples))
            axis.plot(xs, ys, marker="o", label=config)
        axis.set_xlabel("Arrival rate (requests/sec)")
        axis.set_ylabel("Output tokens/sec" if metric.startswith("throughput") else "Seconds")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for config in ("paged_engine", "contiguous_engine"):
        selected = [row for row in rows if row["configuration"] == config]
        if selected:
            axis.scatter(
                [row["max_concurrent_sequences"] for row in selected],
                [row.get("peak_kv_cache_bytes", 0) / (1024**2) for row in selected],
                label=config,
                alpha=0.8,
            )
    axis.set_xlabel("Maximum concurrent sequences")
    axis.set_ylabel("Peak KV cache footprint (MiB)")
    axis.set_title("KV cache footprint vs concurrent sequences")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "memory_vs_concurrency.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--rates", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()
    if args.trials < 3:
        parser.error("--trials must be at least 3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LM.from_huggingface(args.model).to(device).eval()
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    configurations = ["naive_sequential", "static_batching", "contiguous_engine", "paged_engine"]
    rows = []
    for rate_index, rate in enumerate(args.rates):
        trace = generate_trace(
            args.num_requests,
            rate,
            seed=args.seed + rate_index,
            vocab_size=model.config.vocab_size,
            max_prompt_tokens=192,
            max_output_tokens=96,
        )
        for name in configurations:
            warm_trace = _copy_requests(trace[: min(args.batch_size, len(trace))], zero_arrivals=True)

            def run_warmup() -> None:
                run_configuration(
                    name,
                    model,
                    warm_trace,
                    device=device,
                    batch_size=args.batch_size,
                    num_blocks=args.num_blocks,
                    block_size=args.block_size,
                )

            # Warm up every backend before collecting the required three timed trials.
            run_warmup()
            for trial in range(args.trials):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                _sync(device)
                start = time.perf_counter()
                payload = run_configuration(
                    name,
                    model,
                    trace,
                    device=device,
                    batch_size=args.batch_size,
                    num_blocks=args.num_blocks,
                    block_size=args.block_size,
                )
                _sync(device)
                payload["elapsed_seconds"] = time.perf_counter() - start
                peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                row = _summarize(name, payload, peak_memory, args.batch_size)
                row.update(
                    {
                        "configuration": name,
                        "arrival_rate": rate,
                        "trial": trial,
                        "gpu_model": gpu_name,
                        "device": str(device),
                    }
                )
                rows.append(row)
                print(
                    f"{name:20s} rate={rate:g} trial={trial + 1}/{args.trials} "
                    f"throughput={row['throughput_output_tokens_per_second']:.2f} output tok/s"
                )
                del payload
                if device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "project": "mini-vllm",
        "gpu_model": gpu_name,
        "device": str(device),
        "torch_version": torch.__version__,
        "model": args.model,
        "seed": args.seed,
        "trials_per_configuration": args.trials,
        "rows": rows,
        "interpretation": {
            "throughput": "batching effect; compare the naive, static, and continuous configurations",
            "paged_cache": "memory efficiency effect; compare paged_engine with contiguous_engine",
        },
    }
    with (args.output_dir / "benchmark.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    _plot(rows, args.output_dir)
    print(f"Wrote results and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
