"""Small request scheduler with pluggable admission order."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Request:
    request_id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    arrival_time: float = 0.0
    estimated_total_tokens: int | None = None
    generated_tokens: list[int] = field(default_factory=list)
    first_token_time: float | None = None
    finish_time: float | None = None
    prefill_logits: object | None = None

    @property
    def length_estimate(self) -> int:
        if self.estimated_total_tokens is not None:
            return self.estimated_total_tokens
        return len(self.prompt_tokens) + self.max_new_tokens

    @property
    def context_tokens(self) -> list[int]:
        """Tokens needed to rebuild a preempted request's KV state."""
        return self.prompt_tokens + self.generated_tokens


class AdmissionPolicy(Protocol):
    def order(self, requests: list[Request]) -> list[Request]: ...


class FCFS:
    def order(self, requests: list[Request]) -> list[Request]:
        return requests


class ShortestJobFirst:
    """Order requests by an estimated total prompt-plus-output length."""

    def order(self, requests: list[Request]) -> list[Request]:
        return sorted(requests, key=lambda request: request.length_estimate)


class Scheduler:
    def __init__(self, max_batch_size: int, policy: AdmissionPolicy | None = None) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self.policy = policy or FCFS()
        self.waiting: deque[Request] = deque()
        self.running: dict[int, Request] = {}

    def enqueue(self, request: Request) -> None:
        self.waiting.append(request)

    def remove_waiting(self, request: Request) -> None:
        self.waiting.remove(request)

    def start(self, request: Request) -> None:
        self.running[request.request_id] = request

    def finish(self, request_id: int) -> Request:
        return self.running.pop(request_id)
