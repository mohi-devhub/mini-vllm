"""Render the checked-in Molab run summary as README-ready charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs" / "benchmark-results" / "summary.json"
OUTPUT = ROOT / "docs" / "benchmark-results"

COLORS = {
    "naive_sequential": "#64748b",
    "static_batching": "#f59e0b",
    "contiguous_engine": "#2563eb",
    "paged_engine": "#0f9d78",
}
LABELS = {
    "naive_sequential": "Naive sequential",
    "static_batching": "Static batching",
    "contiguous_engine": "Continuous, contiguous KV",
    "paged_engine": "Continuous, paged KV",
}


def _finish(axis, path: Path) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#cbd5e1", alpha=0.45)
    axis.set_axisbelow(True)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(axis.figure)


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rates = data["arrival_rates_requests_per_second"]

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for name, values in data["throughput_output_tokens_per_second"].items():
        axis.plot(
            rates,
            values,
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=COLORS[name],
            label=LABELS[name],
        )
    axis.set(title="Output throughput", xlabel="Arrival rate (requests/s)", ylabel="Output tokens/s")
    axis.set_xticks(rates)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    _finish(axis, OUTPUT / "throughput.png")

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for name, values in data["latency_seconds_p95"].items():
        axis.plot(
            rates,
            [value * 1000 for value in values],
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=COLORS[name],
            label=LABELS[name],
        )
    axis.set_yscale("log")
    axis.set(title="P95 end-to-end latency", xlabel="Arrival rate (requests/s)", ylabel="Milliseconds (log scale)")
    axis.set_xticks(rates)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    _finish(axis, OUTPUT / "p95_latency.png")

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    positions = list(range(len(rates)))
    width = 0.36
    contiguous = data["peak_kv_cache_mib"]["contiguous_engine"]
    paged = data["peak_kv_cache_mib"]["paged_engine"]
    axis.bar(
        [position - width / 2 for position in positions],
        contiguous,
        width,
        color=COLORS["contiguous_engine"],
        label=LABELS["contiguous_engine"],
    )
    axis.bar(
        [position + width / 2 for position in positions],
        paged,
        width,
        color=COLORS["paged_engine"],
        label=LABELS["paged_engine"],
    )
    concurrency = data["max_concurrent_sequences"]
    axis.set(
        title="Peak active KV footprint",
        xlabel="Arrival rate (requests/s); observed concurrency shown below",
        ylabel="Peak KV cache footprint (MiB)",
    )
    axis.set_xticks(positions, [f"{rate} req/s\n{count} concurrent" for rate, count in zip(rates, concurrency)])
    axis.legend(frameon=False, loc="upper left")
    _finish(axis, OUTPUT / "kv_footprint.png")


if __name__ == "__main__":
    main()
