# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from dataclasses import dataclass

import torch

from vllm.v1.drafter.links import Client
from vllm.v1.drafter.runner import StepInput


@dataclass
class Step:
    """A step in the engine, from its request's arrival to its reply."""

    client: Client
    priority: int
    input: StepInput
    # Recorded on the client's stream behind the receive of the payload.
    landed: torch.cuda.Event
    arrived: float
    # When the engine first saw the payload landed.
    landed_at: float | None = None

    @property
    def num_sessions(self) -> int:
        return len(self.input.rows)

    @property
    def num_tokens(self) -> int:
        return self.input.layout.num_tokens


class Scheduler:
    """The pending pool, and which of its steps make the next round.

    A step enters the pool when the engine takes its request and leaves when
    its round is launched. The engine works in rounds as an inference engine
    works in steps: a round drafts for every landed step, up to `max_sessions`
    sessions and `max_num_tokens` tokens, and a step that does not fit waits
    for the next round. A step is one unit, never split over rounds, and a
    client has at most one in the pool (its payloads arrive in order on one
    link).
    """

    def __init__(
        self,
        policy: str,
        max_sessions: int,
        max_num_tokens: int,
        num_speculative_tokens: int,
        batch_window: float = 0.0,
    ):
        self.policy = policy
        self.max_sessions = max_sessions
        self.max_num_tokens = max_num_tokens
        self.k = num_speculative_tokens
        # Seconds a landed step waits for the payloads still on their way.
        self.batch_window = batch_window
        self._ready_since: float | None = None
        # client id -> its step
        self._pool: dict[int, Step] = {}

    def __len__(self) -> int:
        return len(self._pool)

    def __contains__(self, client_id: int) -> bool:
        return client_id in self._pool

    def add(self, step: Step) -> None:
        assert step.client.id not in self._pool
        self._pool[step.client.id] = step

    def pop(self, client_id: int) -> Step | None:
        return self._pool.pop(client_id, None)

    def fits(self, num_sessions: int, num_tokens: int) -> bool:
        """Whether a round can hold this many sessions and tokens."""
        # A draft model adds a slot per session to its first pass; DFlash
        # runs k + 1 query tokens per session.
        return (
            num_sessions <= self.max_sessions
            and num_tokens + num_sessions <= self.max_num_tokens
            and num_sessions * (self.k + 1) <= self.max_num_tokens
        )

    def schedule(self, gpu_free: bool) -> list[Step]:
        """The steps of the next round, taken out of the pool, or none yet.

        A round starts once the GPU is done with the last one, so the steps
        that land meanwhile share the next round.
        """
        now = time.perf_counter()
        ready = []
        for step in self._pool.values():
            if step.landed_at is None and step.landed.query():
                step.landed_at = now
            if step.landed_at is not None:
                ready.append(step)
        if not gpu_free or not ready:
            return []
        if self.batch_window > 0:
            if self._ready_since is None:
                self._ready_since = now
            waited = now - self._ready_since
            if len(ready) < len(self._pool) and waited < self.batch_window:
                return []
            self._ready_since = None

        ready.sort(key=self._order)
        steps: list[Step] = []
        sessions = tokens = 0
        for step in ready:
            more_sessions = sessions + step.num_sessions
            more_tokens = tokens + step.num_tokens
            if not self.fits(more_sessions, more_tokens):
                # Waits for the next round. The first step always fits: the
                # engine checked the widest step of each client at attach.
                assert steps, "a step exceeds what its client attached with"
                continue
            steps.append(step)
            sessions, tokens = more_sessions, more_tokens
        for step in steps:
            del self._pool[step.client.id]
        return steps

    def _order(self, step: Step) -> tuple[int, float]:
        """Sort key of a waiting step: who gets into a round first."""
        if self.policy == "priority":
            return step.priority, step.arrived
        return 0, step.arrived
