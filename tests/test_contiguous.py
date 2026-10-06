import torch

from mini_vllm.baselines import generate_naive, generate_static_batch
from mini_vllm.model import GPT2Config, GPT2LM
from mini_vllm.scheduler import Request


def _tiny_model() -> GPT2LM:
    torch.manual_seed(13)
    model = GPT2LM(GPT2Config(vocab_size=67, max_position_embeddings=32, hidden_size=32, num_layers=2, num_heads=4))
    return model.eval()


def test_contiguous_cached_decode_matches_full_recomputation() -> None:
    model = _tiny_model()
    prompt = [5, 11, 17, 23]
    cached = generate_naive(model, prompt, 7)

    tokens = torch.tensor([prompt])
    expected = []
    with torch.inference_mode():
        for _ in range(7):
            token = int(model(tokens)[0, -1].argmax())
            expected.append(token)
            tokens = torch.cat((tokens, torch.tensor([[token]])), dim=1)
    assert cached == expected


def test_contiguous_cache_capacity_error() -> None:
    model = _tiny_model()
    try:
        generate_naive(model, [1, 2, 3], 4, cache_capacity=5)
    except MemoryError as exc:
        assert "capacity 5" in str(exc)
    else:
        raise AssertionError("expected the cache to reject writes past its capacity")


def test_static_batch_matches_naive_for_mixed_lengths() -> None:
    model = _tiny_model()
    requests = [
        Request(1, [5, 11, 17, 23], 6),
        Request(2, [7, 13], 2),
        Request(3, [9, 15, 21], 4),
    ]
    actual = generate_static_batch(model, requests)
    expected = {
        request.request_id: generate_naive(model, request.prompt_tokens, request.max_new_tokens)
        for request in requests
    }
    assert actual == expected
