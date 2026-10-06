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

The notebook is a thin runner. Update its `REPOSITORY_URL` cell, then run it on a GPU runtime such as a 16 GB T4. It installs the repo, invokes the benchmark script, and displays the generated plots.

## Benchmark results

**Not measured yet.** This checkout has no cached GPT-2 checkpoint or GPU benchmark run. Fill this table and retain the generated plots after running the notebook; do not use illustrative or estimated numbers as results.

| Configuration | Output throughput | TTFT p50 / p95 | E2E latency p50 / p95 / p99 | Peak GPU memory | Max concurrent sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive sequential, contiguous KV | _Run notebook_ | _Run notebook_ | _Run notebook_ | _Run notebook_ | _Run notebook_ |
| Static batch, contiguous KV | _Run notebook_ | _Run notebook_ | _Run notebook_ | _Run notebook_ | _Run notebook_ |
| Continuous batch, contiguous KV | _Run notebook_ | _Run notebook_ | _Run notebook_ | _Run notebook_ | _Run notebook_ |
| Continuous batch, paged KV | _Run notebook_ | _Run notebook_ | _Run notebook_ | _Run notebook_ | _Run notebook_ |

Plots to fill after the notebook run:

- `results/throughput_vs_rate.png` — compare naive, static, and continuous batching to show the batching effect.
- `results/p95_latency_vs_rate.png` — compare tail latency as offered arrival rate rises.
- `results/memory_vs_concurrency.png` — compare resident paged blocks with max-length contiguous reservations at the measured concurrency. Total peak GPU memory is reported separately because the paged pool is preallocated.

Interpret throughput gains by comparing batching strategies. Attribute paging's gain to cache memory efficiency by comparing paged and contiguous continuous batching at the same configured cache budget. Do not describe a paged-vs-contiguous throughput difference as a general batching speedup.

## Limitations and next steps

- Paged attention gathers blocks in PyTorch; it does not use a custom CUDA kernel.
- The engine is synchronous, single-process, and single-GPU.
- Decoding is greedy and fixed-length; EOS is ignored.
- GPT-2 small is the only implemented model, with a 1024-position context limit.
- There is no prefix sharing or copy-on-write.
- The synthetic trace uses oracle output lengths for shortest-job-first; a learned length predictor can replace that estimate later.

Natural next steps are prefix sharing with copy-on-write, sampling, streaming, multi-GPU execution, quantization, support for more model architectures, and a fused paged-attention kernel.
