# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A target's drafter engines as one: `DrafterRouter`.

Same interface as `DrafterClient`, over any number of engines, of any kind:
they only have to draft the same number of tokens. Which engine drafts for a
session is decided when the session opens, by whoever paired the request
with a drafter (`open(session, endpoint)`), or else by load, and stays: that
engine holds the session's KV. `submit()` cuts a step by engine and
`collect()` puts the replies back in the order of the step; the caller never
sees where a session was drafted.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.drafter.client import DrafterClient, PendingDraft
from vllm.v1.drafter.protocol import (
    STEP_FIELDS,
    Attach,
    DrafterInfo,
    DraftRequest,
    StepLayout,
    pack_step,
    unpack_step,
)

logger = init_logger(__name__)


@dataclass
class PendingDrafts:
    """A submitted step: its parts, one per engine that got some of it."""

    num_sessions: int
    # (engine, its part, where the part's sessions are in the step; None:
    # the part is the whole step).
    parts: list[tuple[DrafterClient, PendingDraft, torch.Tensor | None]]


def split_layout(
    layout: StepLayout, positions: list[int]
) -> tuple[StepLayout, np.ndarray]:
    """The layout of the sessions at `positions` of a step, and where their
    tokens are in the step."""
    starts = layout.query_start_loc
    tokens = np.concatenate(
        [np.arange(starts[p], starts[p + 1], dtype=np.int64) for p in positions]
    )
    query_start_loc = np.zeros(len(positions) + 1, dtype=np.int32)
    np.cumsum([starts[p + 1] - starts[p] for p in positions], out=query_start_loc[1:])
    sub = StepLayout(
        num_tokens=len(tokens),
        query_start_loc=query_start_loc.tolist(),
        num_scheduled_tokens=[layout.num_scheduled_tokens[p] for p in positions],
        seq_lens_upper_bound=[layout.seq_lens_upper_bound[p] for p in positions],
        is_prefilling=[layout.is_prefilling[p] for p in positions],
        decode_graph_eligible=layout.decode_graph_eligible,
        max_query_len=layout.max_query_len,
    )
    return sub, tokens


def split_step(
    fields: dict[str, torch.Tensor],
    positions: torch.Tensor,
    tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """The payload (protocol.pack_step) of the sessions at `positions` of a
    step, from the step's unpacked payload."""
    return pack_step(
        fields["input_ids"][tokens],
        fields["positions"][tokens],
        query_start_loc,
        *(fields[name][positions] for name in STEP_FIELDS),
    )


class DrafterRouter:
    def __init__(
        self, endpoints: list[str], device: torch.device, timeout: float = 60.0
    ):
        assert endpoints
        self.device = device
        self.clients = [DrafterClient(e, device, timeout) for e in endpoints]
        self._index = {e: i for i, e in enumerate(endpoints)}
        # session -> index of the engine that holds its KV
        self._engine_of: dict[int, int] = {}
        self._num_sessions = [0] * len(self.clients)
        self._k = 0

    @property
    def num_engines(self) -> int:
        return len(self.clients)

    @property
    def attached(self) -> bool:
        return all(client.attached for client in self.clients)

    def describe(self) -> list[DrafterInfo]:
        """What each engine serves, in the order of the endpoints. Blocks
        until every engine is up."""
        return [client.describe() for client in self.clients]

    def use_hidden_layers(self, default: tuple[int, ...]) -> tuple[int, ...]:
        """Settle the layers each engine gets hidden states of (see
        DrafterClient.use_hidden_layers). Returns every layer some engine
        reads: what `submit()` must be given."""
        layers: set[int] = set()
        for client in self.clients:
            layers.update(client.use_hidden_layers(default))
        return tuple(sorted(layers))

    def attach(self, attach: Attach) -> None:
        for client in self.clients:
            if not client.attached:
                client.attach(attach)
        self._k = attach.num_speculative_tokens

    def detach(self) -> None:
        for client in self.clients:
            client.detach()

    def open(self, session: int, endpoint: str | None = None) -> None:
        """A new session. `endpoint` names the engine that drafts for it;
        without one, or with one this router does not have, the session goes
        to the engine that has the fewest."""
        self.release([session])
        engine = self._index.get(endpoint) if endpoint is not None else None
        if endpoint is not None and engine is None:
            logger.warning_once(
                "Drafter %s is not one of this target's engines (%s); "
                "the request is drafted by another one.",
                endpoint,
                ", ".join(self._index),
            )
        if engine is None:
            engine = min(range(len(self.clients)), key=self._num_sessions.__getitem__)
        self._engine_of[session] = engine
        self._num_sessions[engine] += 1

    def release(self, sessions: list[int]) -> None:
        released: dict[int, list[int]] = {}
        for session in sessions:
            engine = self._engine_of.pop(session, None)
            if engine is not None:
                self._num_sessions[engine] -= 1
                released.setdefault(engine, []).append(session)
        for engine, engine_sessions in released.items():
            self.clients[engine].release(engine_sessions)

    def submit(
        self,
        request: DraftRequest,
        step: torch.Tensor,
        hidden: Mapping[int, torch.Tensor] | None = None,
    ) -> PendingDrafts:
        """Send one step to the engines of its sessions; see
        DrafterClient.submit.

        A step that goes to several engines is cut by session, which needs
        its exact layout on the CPU: for a step the target sized on the GPU
        (adaptive verification), the caller has to read the sizes back first
        (`layout.exact_starts`).
        """
        b, layout = len(request.sessions), request.layout
        groups = self._route(request.sessions)
        if len(groups) == 1:
            client = self.clients[next(iter(groups))]
            return PendingDrafts(
                b, [(client, client.submit(request, step, hidden), None)]
            )
        if not layout.exact_starts:
            raise ValueError(
                "a step sized on the GPU can only be cut by engine with its "
                "exact layout (StepLayout.exact_starts)"
            )

        fields = unpack_step(step, b, layout.num_tokens)
        parts = []
        for engine, positions in groups.items():
            client = self.clients[engine]
            sub_layout, tokens = split_layout(layout, positions)
            sessions = [request.sessions[p] for p in positions]
            sub_request = DraftRequest(
                sessions=sessions,
                layout=sub_layout,
                priority=request.priority,
                prefixes={
                    s: request.prefixes[s] for s in sessions if s in request.prefixes
                },
                num_draft_tokens=[
                    request.num_draft_tokens[p]
                    for p in positions
                    if request.num_draft_tokens
                ],
            )
            # No copy may wait for the stream: it is busy with the target.
            positions_t = self._to_device(np.asarray(positions, dtype=np.int64))
            tokens_t = self._to_device(tokens)
            query_start_loc = self._to_device(
                np.asarray(sub_layout.query_start_loc, dtype=np.int32)
            )
            sub_step = split_step(fields, positions_t, tokens_t, query_start_loc)
            sub_hidden = None
            if client.hidden_layers:
                assert hidden is not None, "this engine reads hidden states"
                sub_hidden = {
                    layer: hidden[layer][tokens_t] for layer in client.hidden_layers
                }
            parts.append(
                (client, client.submit(sub_request, sub_step, sub_hidden), positions_t)
            )
        return PendingDrafts(b, parts)

    def collect(self, pending: PendingDrafts) -> torch.Tensor:
        """The reply to a submitted step, in the order of its sessions; see
        DrafterClient.collect."""
        client, part, positions = pending.parts[0]
        if positions is None:
            return client.collect(part)
        reply = torch.empty(
            (pending.num_sessions, 2 * self._k), dtype=torch.int32, device=self.device
        )
        for client, part, positions in pending.parts:
            reply[positions] = client.collect(part)
        return reply

    def _route(self, sessions: list[int]) -> dict[int, list[int]]:
        """Where each engine's sessions are in `sessions`, by engine index."""
        groups: dict[int, list[int]] = {}
        for position, session in enumerate(sessions):
            if session not in self._engine_of:
                self.open(session)
            groups.setdefault(self._engine_of[session], []).append(position)
        return groups

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        return async_tensor_h2d(array, device=self.device)
