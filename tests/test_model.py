import pytest
import torch
from transformers import GPT2Config as HFGPT2Config
from transformers import GPT2LMHeadModel

from mini_vllm.model import GPT2Config, GPT2LM


def _copy_hf_weights(model: GPT2LM, reference: GPT2LMHeadModel) -> None:
    state = reference.state_dict()
    with torch.no_grad():
        model.wte.weight.copy_(state["transformer.wte.weight"])
        model.wpe.weight.copy_(state["transformer.wpe.weight"])
        model.ln_f.weight.copy_(state["transformer.ln_f.weight"])
        model.ln_f.bias.copy_(state["transformer.ln_f.bias"])
        for i, block in enumerate(model.blocks):
            prefix = f"transformer.h.{i}."
            for name in ("ln_1", "ln_2"):
                getattr(block, name).weight.copy_(state[prefix + name + ".weight"])
                getattr(block, name).bias.copy_(state[prefix + name + ".bias"])
            for own, hf in (
                (block.attn.qkv, "attn.c_attn"),
                (block.attn.proj, "attn.c_proj"),
                (block.mlp.fc, "mlp.c_fc"),
                (block.mlp.proj, "mlp.c_proj"),
            ):
                own.weight.copy_(state[prefix + hf + ".weight"].T)
                own.bias.copy_(state[prefix + hf + ".bias"])


def _compare(model: GPT2LM, reference: GPT2LMHeadModel) -> None:
    model.eval()
    reference.eval()
    # Right padded prompts of different lengths exercise both causal and padding masks.
    input_ids = torch.tensor(
        [
            [11, 29, 57, 91, 13, 0, 0, 0, 0],
            [42, 78, 0, 0, 0, 0, 0, 0, 0],
            [3, 7, 9, 17, 28, 31, 44, 55, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
        ]
    )
    with torch.inference_mode():
        actual = model(input_ids, attention_mask)
        expected = reference(input_ids, attention_mask=attention_mask).logits
    torch.testing.assert_close(actual[attention_mask.bool()], expected[attention_mask.bool()], rtol=1e-4, atol=1e-4)


def test_parity_with_huggingface_tiny_random_config() -> None:
    torch.manual_seed(7)
    hf_config = HFGPT2Config(
        vocab_size=101,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    reference = GPT2LMHeadModel(hf_config).eval()
    model = GPT2LM(
        GPT2Config(vocab_size=101, max_position_embeddings=32, hidden_size=32, num_layers=2, num_heads=4)
    )
    _copy_hf_weights(model, reference)
    _compare(model, reference)


def test_parity_with_pretrained_gpt2() -> None:
    try:
        reference = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True).eval()
    except (OSError, ValueError) as exc:
        pytest.skip(f"pretrained GPT-2 weights are not cached locally: {exc}")
    model = GPT2LM.from_huggingface("gpt2").eval()
    _compare(model, reference)
