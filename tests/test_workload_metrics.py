import pytest

from mini_vllm.metrics import percentile
from mini_vllm.workload import generate_trace


def test_synthetic_trace_is_seeded_and_has_poisson_arrivals() -> None:
    first = generate_trace(20, 3.0, seed=8, vocab_size=101, max_prompt_tokens=32, max_output_tokens=16)
    second = generate_trace(20, 3.0, seed=8, vocab_size=101, max_prompt_tokens=32, max_output_tokens=16)
    assert [(r.arrival_time, r.prompt_tokens, r.max_new_tokens) for r in first] == [
        (r.arrival_time, r.prompt_tokens, r.max_new_tokens) for r in second
    ]
    assert all(first[i].arrival_time < first[i + 1].arrival_time for i in range(len(first) - 1))
    assert max(len(request.prompt_tokens) for request in first) > min(len(request.prompt_tokens) for request in first)
    assert max(request.max_new_tokens for request in first) > min(request.max_new_tokens for request in first)


def test_percentile_interpolates_and_handles_empty_input() -> None:
    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([], 95) == 0.0


def test_trace_rejects_invalid_arrival_rate() -> None:
    with pytest.raises(ValueError, match="arrival_rate"):
        generate_trace(2, 0.0)
