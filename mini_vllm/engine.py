"""Synchronous continuous-batching inference engine."""

from __future__ import annotations

import time
from collections.abc import Iterable

import torch
from torch import Tensor

from mini_vllm.cache import BlockPool, ContiguousCachePool, PagedKVCache
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
        cache_mode: str = "paged",
    ) -> None:
        if watermark_blocks < 0:
            raise ValueError("watermark_blocks cannot be negative")
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        if cache_mode not in {"paged", "contiguous"}:
            raise ValueError("cache_mode must be 'paged' or 'contiguous'")
        self.model = model.eval()
        self.cache_mode = cache_mode
        if cache_mode == "paged":
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
        else:
            self.pool = None
            self.cache = ContiguousCachePool(
                model.config.num_layers,
                num_blocks * block_size,
                model.config.num_heads,
                model.config.head_dim,
                device=model.wte.weight.device,
                dtype=model.wte.weight.dtype,
            )
        self.scheduler = Scheduler(max_batch_size, policy)
        self.block_size = block_size
        self.watermark_blocks = watermark_blocks
        self.preemptions = 0
        self.completed: dict[int, Request] = {}
        self.max_concurrent_sequences = 0
        self._clock_origin: float | None = None

    def _prompt_blocks(self, request: Request) -> int:
        length = len(request.context_tokens)
        return (length + self.block_size - 1) // self.block_size

    def _admit(self) -> list[Request]:
        admitted = []
        while self.scheduler.waiting and len(self.scheduler.running) < self.scheduler.max_batch_size:
            ordered = self.scheduler.policy.order(list(self.scheduler.waiting))
            request = None
            for candidate in ordered:
                if self.cache_mode == "paged":
                    fits = self._prompt_blocks(candidate) + self.watermark_blocks <= self.pool.free_blocks
                else:
                    fits = (
                        len(candidate.prompt_tokens)
                        + candidate.max_new_tokens
                        + self.watermark_blocks * self.block_size
                        <= self.cache.free_tokens
                    )
                if fits:
                    request = candidate
                    break
                if not getattr(self.scheduler.policy, "can_bypass_blocked_request", False):
                    break
            if request is None:
                break
            self.scheduler.remove_waiting(request)
            self.scheduler.start(request)
            if self.cache_mode == "paged":
                self.cache.add_sequence(request.request_id)
            else:
                self.cache.add_sequence(
                    request.request_id,
                    len(request.prompt_tokens) + request.max_new_tokens,
                )
            admitted.append(request)
        return admitted

    def _prefill(self, requests: list[Request]) -> None:
        if not requests:
            return
        device = self.model.wte.weight.device
        max_prompt = max(len(request.context_tokens) for request in requests)
        input_ids = torch.zeros((len(requests), max_prompt), dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for row, request in enumerate(requests):
            tokens = torch.tensor(request.context_tokens, dtype=torch.long, device=device)
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
            request.prefill_logits = logits[row, len(request.context_tokens) - 1]

    def _preempt_latest(self) -> None:
        if not self.scheduler.running:
            raise MemoryError("paged KV block pool exhausted with no running request available to preempt")
        request_id = next(reversed(self.scheduler.running))
        request = self.scheduler.finish(request_id)
        self.cache.free_sequence(request_id)
        request.prefill_logits = None
        self.scheduler.waiting.appendleft(request)
        self.preemptions += 1

    def _decode(self, now: float) -> None:
        while True:
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
            if not to_forward:
                break
            last_tokens = torch.tensor(
                [[request.generated_tokens[-1]] for request in to_forward], dtype=torch.long, device=device
            )
            positions = torch.tensor(
                [[len(request.prompt_tokens) + len(request.generated_tokens) - 1] for request in to_forward],
                dtype=torch.long,
                device=device,
            )
            try:
                with torch.inference_mode():
                    logits = self.model(
                        last_tokens,
                        position_ids=positions,
                        cache=self.cache,
                        seq_ids=[request.request_id for request in to_forward],
                    )[:, -1]
            except MemoryError:
                # Restart the batch after freeing the most recently admitted sequence.
                self._preempt_latest()
                continue
            logits_by_id.update({request.request_id: logits[row] for row, request in enumerate(to_forward)})
            break

        sampled_tokens = []
        for request in running:
            token = int(logits_by_id[request.request_id].argmax().item())
            emitted_at = time.perf_counter() - self._clock_origin if self._clock_origin is not None else now
            sampled_tokens.append((request, token, emitted_at))
        finished = []
        for request, token, emitted_at in sampled_tokens:
            request.prefill_logits = None
            request.generated_tokens.append(token)
            if request.first_token_time is None:
                request.first_token_time = emitted_at - request.arrival_time
            if len(request.generated_tokens) >= request.max_new_tokens:
                request.finish_time = emitted_at - request.arrival_time
                finished.append(request.request_id)

        for request_id in finished:
            request = self.scheduler.finish(request_id)
            self.cache.free_sequence(request_id)
            self.completed[request_id] = request

    def step(self, now: float = 0.0) -> bool:
        """Admit, prefill, and emit one token for every currently running request."""
        admitted = self._admit()
        self.max_concurrent_sequences = max(self.max_concurrent_sequences, len(self.scheduler.running))
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
        self._clock_origin = start
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
