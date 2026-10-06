# mini-vLLM

mini-vLLM is a small, readable inference engine built in PyTorch. It implements GPT-2 small (124M parameters), greedy decoding, continuous batching, and a paged KV cache. It is intended to make the serving mechanics easy to inspect, not to compete with a fused production runtime.

The implementation uses Hugging Face only to load GPT-2 weights and as a parity reference in tests. Generation, attention, caching, admission, and scheduling are implemented in this repository. The naive baseline also uses a KV cache, so its decode path does not repeatedly recompute the prompt.

## Layout

```text
mini_vllm/
  model.py       GPT-2 and the cache interface
  cache.py       contiguous caches, block pool, and paged cache
  scheduler.py   requests and FCFS / shortest-job-first policies
  engine.py      synchronous continuous-batching loop
  baselines.py   naive sequential and static batching
  workload.py    seeded Poisson arrivals and heavy-tailed lengths
  metrics.py     timing helpers and summaries
benchmarks/run.py
notebooks/benchmark.ipynb
tests/
```

## How paging works

The pool is allocated once per engine. Its shape is `[layers, K/V, physical blocks, tokens per block, heads, head dimension]`. A physical block ID names the same token slot across every layer and for both K and V.

```text
Logical positions for request 17:
  0 ... 15       16 ... 31       32 ... 47
     table[0]       table[1]        table[2]
        |               |               |
        v               v               v
  +-----------+   +-----------+   +-----------+
  | block  6  |   | block  2  |   | block 11  |
  +-----------+   +-----------+   +-----------+
       \               |               /
        +--------------+--------------+
                       v
Pool: [layer, K/V, physical block, token, head, head_dim]
```

Each sequence owns a logical block table. The allocator gives it physical blocks as positions are written and returns those blocks as soon as the request finishes. The engine tracks peak blocks in use and unused token slots inside allocated blocks.

Paged attention gathers each sequence's K/V blocks into a padded batch and applies a length mask. **This is a PyTorch emulation of the block-table gather a custom paged-attention CUDA kernel would perform.** It is useful for understanding the indirection and testing correctness, but it is not the fused kernel used by vLLM.

## How a step works

1. Newly arrived requests enter the waiting queue. The configured admission policy orders them; FCFS is the default, and shortest-job-first uses each request's estimated total length.
2. The engine admits requests while the running batch has room and the cache can hold each prompt plus its watermark reserve.
3. New prompts are right-padded into one prefill batch. Each sequence gets its own position IDs, and valid K/V positions are written through the cache interface.
4. The first generated token comes from the prefill logits. On later steps, the engine feeds the last generated token for every running request and emits one greedy token per request.
5. Finished requests release their blocks immediately. If a decode write runs out of blocks, the most recently admitted request is preempted, its blocks are freed, and it returns to the front of the queue. Its prompt and generated prefix are recomputed when it is admitted again.

The same GPT-2 attention code accepts either the contiguous cache or the paged cache. GPT-2 absolute positions are computed independently for each sequence, including mixed-length batches. Requests have fixed output lengths and ignore EOS, making traces reproducible.

## Run it

Python 3.10 or newer is required. Install the package and optional benchmark/test tools:

```bash
python -m pip install -e '.[benchmark,test]'
pytest
```

The tests use small random models for fast CPU checks. The pretrained GPT-2 parity test runs when the checkpoint is available in the local Hugging Face cache; it skips when the weights are not cached. The benchmark runner downloads GPT-2 weights if needed.

Run the benchmark locally or in a single-GPU notebook. The device is selected automatically:

```bash
python benchmarks/run.py \
  --num-requests 24 \
  --trials 3 \
  --rates 2 4 8 \
  --batch-size 8 \
  --block-size 16 \
  --num-blocks 512 \
  --output-dir results
```

Each configuration is warmed up and measured for at least three trials. CUDA synchronization brackets each timing. The JSON records the device and GPU model, throughput, first-token and end-to-end latency percentiles, peak GPU memory, peak concurrent sequences, preemptions, and cache statistics. The runner writes:

```text
results/benchmark.json
results/throughput_vs_rate.png
results/p95_latency_vs_rate.png
results/memory_vs_concurrency.png
```

The notebook is a thin runner configured to clone `https://github.com/mohi-devhub/mini-vllm.git`. Run it on a GPU runtime such as a 16 GB T4. It installs the repo, invokes the benchmark script, and displays the generated plots. The notebook uses plain Python subprocess calls so it works in notebook hosts that do not support IPython shell magics.

## Benchmark results

Measured on an NVIDIA RTX PRO 6000 Blackwell Server Edition with PyTorch 2.11.0+cu130, GPT-2, 24 requests per trace, batch size 8, block size 16, and a 512-block pool. Each cell below is the median of three trials. Latencies are milliseconds, peak GPU memory is MiB, and throughput is output tokens/second. TTFT and end-to-end percentiles are the medians of the corresponding per-trial percentile values.

| Arrival rate | Configuration | Throughput | TTFT p50 / p95 | E2E p50 / p95 / p99 | Peak GPU memory | Max concurrent |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 req/s | Naive sequential | 33.95 | 8.34 / 163.24 | 135.99 / 482.28 / 509.87 | 538.0 | 1 |
| 2 req/s | Static batch | 32.03 | 2306.14 / 5547.67 | 2785.05 / 6058.85 / 7014.59 | 820.8 | 8 |
| 2 req/s | Continuous, contiguous KV | 33.97 | 8.41 / 18.04 | 107.40 / 528.01 / 569.81 | 538.0 | 2 |
| 2 req/s | Continuous, paged KV | 33.95 | 11.25 / 25.08 | 109.39 / 544.46 / 580.47 | 1099.2 | 2 |
| 4 req/s | Naive sequential | 45.67 | 7.97 / 74.00 | 51.54 / 181.88 / 217.31 | 496.8 | 1 |
| 4 req/s | Static batch | 43.01 | 811.21 / 1617.04 | 940.84 / 1749.53 / 2062.14 | 560.1 | 8 |
| 4 req/s | Continuous, contiguous KV | 45.64 | 9.05 / 20.25 | 57.17 / 188.20 / 226.96 | 498.1 | 3 |
| 4 req/s | Continuous, paged KV | 45.62 | 11.37 / 26.96 | 61.99 / 192.73 / 245.01 | 1068.7 | 3 |
| 8 req/s | Naive sequential | 98.99 | 14.31 / 405.17 | 133.50 / 508.01 / 542.63 | 537.8 | 1 |
| 8 req/s | Static batch | 90.90 | 346.99 / 1070.58 | 674.04 / 1328.31 / 1529.25 | 823.9 | 8 |
| 8 req/s | Continuous, contiguous KV | 106.62 | 9.89 / 33.68 | 129.76 / 450.14 / 582.41 | 551.2 | 5 |
| 8 req/s | Continuous, paged KV | 102.31 | 13.62 / 41.99 | 136.19 / 545.91 / 702.39 | 1099.2 | 5 |

Peak active/reserved KV footprint from the same trials:

| Arrival rate | Contiguous max-length reservation | Paged resident blocks | Paged peak block utilization | Preemptions |
| ---: | ---: | ---: | ---: | ---: |
| 2 req/s | 14.84 MiB | 15.75 MiB | 2.73% | 0 |
| 4 req/s | 6.75 MiB | 7.88 MiB | 1.37% | 0 |
| 8 req/s | 25.25 MiB | 24.75 MiB | 4.30% | 0 |

The results show little throughput difference between naive sequential and continuous batching at 2 and 4 requests/sec. At 8 requests/sec, continuous contiguous KV reached 106.62 output tokens/sec versus 98.99 for naive sequential; paged KV reached 102.31. Static batching was slower at all three rates and had much higher TTFT because it waits for groups of eight requests before starting a batch.

This run does not demonstrate a concurrency or GPU-memory advantage for paging. The paged and contiguous engines both reached 2, 3, and 5 concurrent sequences at the three rates, with no preemptions. The 512-block pool reached only 23 blocks in use at peak, so this workload did not create meaningful memory pressure. The paged pool is preallocated, which explains its higher total GPU-memory reading; compare the KV footprint rows separately from total GPU memory. A smaller cache budget or longer requests is needed to measure the concurrency benefit under pressure.

Plots from this run are in `results/throughput_vs_rate.png`, `results/p95_latency_vs_rate.png`, and `results/memory_vs_concurrency.png`. Throughput differences between paged and contiguous KV should not be presented as the batching gain; this run shows paging's gather overhead and no measurable capacity gain at the configured budget.

## Limitations and next steps

- Paged attention gathers blocks in PyTorch; it does not use a custom CUDA kernel.
- The engine is synchronous, single-process, and single-GPU.
- Decoding is greedy and fixed-length; EOS is ignored.
- GPT-2 small is the only implemented model, with a 1024-position context limit.
- There is no prefix sharing or copy-on-write.
- The synthetic trace uses oracle output lengths for shortest-job-first; a learned length predictor can replace that estimate later.

Natural next steps are prefix sharing with copy-on-write, sampling, streaming, multi-GPU execution, quantization, support for more model architectures, and a fused paged-attention kernel.
