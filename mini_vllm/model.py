"""A small GPT-2 implementation with a pluggable key/value cache."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int = 50_257
    max_position_embeddings: int = 1_024
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class KVCache(Protocol):
    """Storage interface used by attention; caches are owned by the caller."""

    def write(
        self,
        layer_idx: int,
        seq_id: int,
        positions: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> None: ...

    def read(self, layer_idx: int, seq_id: int, end_position: int) -> tuple[Tensor, Tensor]: ...


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        cache: KVCache | None = None,
        seq_ids: Sequence[int] | None = None,
    ) -> Tensor:
        batch, query_len, hidden = x.shape
        qkv = self.qkv(x).view(batch, query_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if cache is None:
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            q_positions = position_ids[:, :, None]
            k_positions = position_ids[:, None, :]
            allowed = (k_positions <= q_positions) & attention_mask[:, None, :].bool()
            scores = scores.masked_fill(~allowed[:, None, :, :], torch.finfo(scores.dtype).min)
            weights = self.attn_dropout(torch.softmax(scores, dim=-1))
            attended = weights @ value
        else:
            if seq_ids is None or len(seq_ids) != batch:
                raise ValueError("one seq_id is required for each cached sequence")
            attended_rows = []
            for row, seq_id in enumerate(seq_ids):
                valid = attention_mask[row].bool()
                row_positions = position_ids[row, valid]
                if not row_positions.numel():
                    attended_rows.append(x.new_zeros((self.num_heads, query_len, self.head_dim)))
                    continue
                row_key = key[row, :, valid, :]
                row_value = value[row, :, valid, :]
                cache.write(self.layer_idx, seq_id, row_positions, row_key, row_value)
                end_position = int(row_positions.max().item()) + 1
                all_key, all_value = cache.read(self.layer_idx, seq_id, end_position)
                row_query = query[row, :, valid, :]
                scores = row_query @ all_key.transpose(-2, -1) / math.sqrt(self.head_dim)
                allowed = torch.arange(end_position, device=x.device)[None, :] <= row_positions[:, None]
                scores = scores.masked_fill(~allowed[None, :, :], torch.finfo(scores.dtype).min)
                weights = self.attn_dropout(torch.softmax(scores, dim=-1))
                row_attended = weights @ all_value
                padded = x.new_zeros((self.num_heads, query_len, self.head_dim))
                padded[:, valid, :] = row_attended
                attended_rows.append(padded)
            attended = torch.stack(attended_rows)

        attended = attended.transpose(1, 2).contiguous().view(batch, query_len, hidden)
        return self.resid_dropout(self.proj(attended))


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.proj = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        # GPT-2's NewGELU activation, rather than PyTorch's exact GELU variant.
        x = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
        return self.dropout(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        cache: KVCache | None,
        seq_ids: Sequence[int] | None,
    ) -> Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask, position_ids, cache, seq_ids)
        return x + self.mlp(self.ln_2(x))


class GPT2LM(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config, i) for i in range(config.num_layers))
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cache: KVCache | None = None,
        seq_ids: Sequence[int] | None = None,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.clamp_min(0)
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids shape")
        if int(position_ids.max().item()) >= self.config.max_position_embeddings:
            raise ValueError("position index exceeds the model's positional embedding table")

        x = self.drop(self.wte(input_ids) + self.wpe(position_ids))
        for block in self.blocks:
            x = block(x, attention_mask, position_ids, cache, seq_ids)
        return self.lm_head(self.ln_f(x))

    @classmethod
    def from_huggingface(cls, model_name: str = "gpt2") -> "GPT2LM":
        """Load GPT-2 weights, transposing Hugging Face Conv1D matrices."""
        from transformers import GPT2LMHeadModel

        reference = GPT2LMHeadModel.from_pretrained(model_name)
        hf_config = reference.config
        config = GPT2Config(
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.n_positions,
            hidden_size=hf_config.n_embd,
            num_layers=hf_config.n_layer,
            num_heads=hf_config.n_head,
            dropout=0.0,
        )
        model = cls(config)
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
        model.eval()
        return model
