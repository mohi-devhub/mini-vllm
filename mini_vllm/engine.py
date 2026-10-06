"""Synchronous continuous-batching inference engine."""

from __future__ import annotations

import time
from collections.abc import Iterable

import torch
from torch import Tensor

from mini_vllm.cache import BlockPool, PagedKVCache
from mini_vllm.model import GPT2LM
from mini_vllm.scheduler import AdmissionPolicy, Request, Scheduler


class Engine:
    def __init__(
        self,
        model: GPT2LM,
        *,
        num_blocks: int,
        block_size: int = 16,
        max_batch_size: int = 8,
        watermark_blocks: int = 1,
        policy: AdmissionPolicy | None = None,
    ) -> None:
        if watermark_blocks < 0:
            raise ValueError("watermark_blocks cannot be negative")
        self.model = model.eval()
        self.pool = BlockPool(
            model.config.num_layers,
            num_blocks,
            block_size,
            model.config.num_heads,
            model.config.head_dim,
            device=model.wte.weight.device,
            dtype=model.wte.weight.dtype,
        )
        self.cache = PagedKVCache(self.pool)
        self.scheduler = Scheduler(max_batch_size, policy)
        self.block_size = block_size
        self.watermark_blocks = watermark_blocks
        self.preemptions = 0
        self.completed: dict[int, Request] = {}

    def _prompt_blocks(self, request: Request) -> int:
        return (len(request.prompt_tokens) + self.block_size - 1) // self.block_size

    def _admit(self) -> list[Request]:
        admitted = []
        while self.scheduler.waiting and len(self.scheduler.running) < self.scheduler.max_batch_size:
            ordered = self.scheduler.policy.order(list(self.scheduler.waiting))
            request = next(
                (
                    candidate
                    for candidate in ordered
                    if self._prompt_blocks(candidate) + self.watermark_blocks <= self.pool.free_blocks
                ),
                None,
            )
            if request is None:
                break
            self.scheduler.remove_waiting(request)
            self.scheduler.start(request)
            self.cache.add_sequence(request.request_id)
            admitted.append(request)
        return admitted

    def _prefill(self, requests: list[Request]) -> None:
        if not requests:
            return
        device = self.model.wte.weight.device
        max_prompt = max(len(request.prompt_tokens) for request in requests)
        input_ids = torch.zeros((len(requests), max_prompt), dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for row, request in enumerate(requests):
            tokens = torch.tensor(request.prompt_tokens, dtype=torch.long, device=device)
            input_ids[row, : tokens.numel()] = tokens
            attention_mask[row, : tokens.numel()] = 1
        position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=self.cache,
                seq_ids=[request.request_id for request in requests],
            )
        for row, request in enumerate(requests):
            request.prefill_logits = logits[row, len(request.prompt_tokens) - 1]

    def _decode(self, now: float) -> None:
        running = list(self.scheduler.running.values())
        if not running:
            return
        device = self.model.wte.weight.device
        to_forward = [request for request in running if request.prefill_logits is None]
        logits_by_id: dict[int, Tensor] = {
            request.request_id: request.prefill_logits
            for request in running
            if request.prefill_logits is not None
        }
        if to_forward:
            last_tokens = torch.tensor(
                [[request.generated_tokens[-1]] for request in to_forward], dtype=torch.long, device=device
            )
            positions = torch.tensor(
                [[len(request.prompt_tokens) + len(request.generated_tokens) - 1] for request in to_forward],
                dtype=torch.long,
                device=device,
            )
            with torch.inference_mode():
                logits = self.model(
                    last_tokens,
                    position_ids=positions,
                    cache=self.cache,
                    seq_ids=[request.request_id for request in to_forward],
                )[:, -1]
            logits_by_id.update({request.request_id: logits[row] for row, request in enumerate(to_forward)})

        finished = []
        for request in running:
            token = int(logits_by_id[request.request_id].argmax().item())
            request.prefill_logits = None
            request.generated_tokens.append(token)
            if request.first_token_time is None:
                request.first_token_time = now - request.arrival_time
            if len(request.generated_tokens) >= request.max_new_tokens:
                request.finish_time = now - request.arrival_time
                finished.append(request.request_id)

        for request_id in finished:
            request = self.scheduler.finish(request_id)
            self.cache.free_sequence(request_id)
            self.completed[request_id] = request

    def step(self, now: float = 0.0) -> bool:
        """Admit, prefill, and emit one token for every currently running request."""
        admitted = self._admit()
        self._prefill(admitted)
        had_running = bool(self.scheduler.running)
        self._decode(now)
        return bool(admitted) or had_running

    def run(self, requests: Iterable[Request]) -> dict[int, list[int]]:
        """Replay relative arrival times against a monotonic wall clock."""
        pending = sorted(requests, key=lambda request: request.arrival_time)
        if len({request.request_id for request in pending}) != len(pending):
            raise ValueError("request IDs must be unique")
        if any(not request.prompt_tokens for request in pending):
            raise ValueError("every request needs a non-empty prompt")
        if any(request.max_new_tokens <= 0 for request in pending):
            raise ValueError("max_new_tokens must be positive")
        if any(len(r.prompt_tokens) + r.max_new_tokens > self.model.config.max_position_embeddings for r in pending):
            raise ValueError("prompt plus output exceeds model position embeddings")
        start = time.perf_counter()
        cursor = 0
        while cursor < len(pending) or self.scheduler.waiting or self.scheduler.running:
            elapsed = time.perf_counter() - start
            while cursor < len(pending) and pending[cursor].arrival_time <= elapsed:
                self.scheduler.enqueue(pending[cursor])
                cursor += 1
            if self.scheduler.waiting or self.scheduler.running:
                made_progress = self.step(time.perf_counter() - start)
                if not made_progress and not self.scheduler.running:
                    request = self.scheduler.waiting[0]
                    raise MemoryError(
                        f"prompt for request {request.request_id} cannot fit in the available paged KV pool"
                    )
                continue
            if cursor < len(pending):
                time.sleep(max(0.0, pending[cursor].arrival_time - (time.perf_counter() - start)))
        return {request_id: request.generated_tokens for request_id, request in self.completed.items()}
