import random

import torch

from mini_vllm.baselines import generate_naive
from mini_vllm.engine import Engine
from mini_vllm.model import GPT2Config, GPT2LM
from mini_vllm.scheduler import Request, ShortestJobFirst


def _tiny_model() -> GPT2LM:
    torch.manual_seed(31)
    return GPT2LM(
        GPT2Config(vocab_size=73, max_position_embeddings=64, hidden_size=32, num_layers=2, num_heads=4)
    ).eval()


def test_engine_matches_naive_for_randomized_trace() -> None:
    model = _tiny_model()
    rng = random.Random(44)
    requests = [
        Request(
            request_id=i,
            prompt_tokens=[rng.randrange(1, 73) for _ in range(rng.randint(2, 11))],
            max_new_tokens=rng.randint(1, 7),
            arrival_time=0.0,
        )
        for i in range(14)
    ]
    rng.shuffle(requests)
    expected = {
        request.request_id: generate_naive(model, request.prompt_tokens, request.max_new_tokens)
        for request in requests
    }
    engine = Engine(model, num_blocks=48, block_size=4, max_batch_size=4, watermark_blocks=1)
    actual = engine.run(requests)

    for request_id, expected_tokens in expected.items():
        assert actual[request_id] == expected_tokens, f"request {request_id}: {actual[request_id]} != {expected_tokens}"
    assert engine.pool.blocks_in_use == 0
    assert engine.pool.peak_blocks > 0


def test_shortest_job_first_uses_estimated_total_length() -> None:
    model = _tiny_model()
    requests = [
        Request(1, [1, 2], 3, estimated_total_tokens=50),
        Request(2, [1, 2], 3, estimated_total_tokens=7),
    ]
    engine = Engine(model, num_blocks=16, block_size=4, max_batch_size=1, watermark_blocks=0, policy=ShortestJobFirst())
    engine.scheduler.enqueue(requests[0])
    engine.scheduler.enqueue(requests[1])
    admitted = engine._admit()
    assert [request.request_id for request in admitted] == [2]


def test_preemption_recomputes_state_and_preserves_outputs() -> None:
    model = _tiny_model()
    requests = [
        Request(10, [3, 9, 15, 21], 7),
        Request(20, [4, 10, 16, 22], 7),
    ]
    expected = {
        request.request_id: generate_naive(model, request.prompt_tokens, request.max_new_tokens)
        for request in requests
    }
    engine = Engine(model, num_blocks=4, block_size=4, max_batch_size=2, watermark_blocks=0)
    actual = engine.run(requests)
    assert engine.preemptions > 0
    assert actual == expected
    assert engine.pool.blocks_in_use == 0


def test_contiguous_continuous_engine_matches_naive() -> None:
    model = _tiny_model()
    requests = [Request(31, [3, 9, 15], 4), Request(32, [4, 10], 6)]
    expected = {
        request.request_id: generate_naive(model, request.prompt_tokens, request.max_new_tokens)
        for request in requests
    }
    engine = Engine(
        model,
        num_blocks=16,
        block_size=4,
        max_batch_size=2,
        cache_mode="contiguous",
    )
    assert engine.run(requests) == expected
