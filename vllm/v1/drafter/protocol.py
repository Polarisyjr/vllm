# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire protocol of the drafter engine.

A client is a target engine. It attaches at any time, sends one `DraftRequest`
per target step and detaches when it is done. Nothing is reserved for a
client: a session takes a row and KV blocks when it first shows up in a
request and returns them on `Release`, or when the client goes away.

The interface is the same for every drafter. A target sends the tokens of its
step and their verify result, plus the hidden states of the layers the engine
names in its `DrafterInfo`, and gets k draft tokens and their acceptance
confidences back. What kind of drafter turned one into the other is the
engine's business, so one target may use engines of different kinds at once.

k is fixed, how much of it is drafted is not: a session may ask for fewer
tokens (`DraftRequest.num_draft_tokens`) and an engine may be told to draft
fewer for everyone (`SetDraftTokens`). The positions past that are padding,
with confidence 0, which adaptive verification does not spend tokens on.

A session is one sequence the drafter keeps KV for, named by the client (the
request's slot in the target's request state). A request replays the target's
step for its sessions: the tokens the target just ran (prompt chunks, or the
previous drafts), the verify result, and for heads the target's hidden states.
The drafter keeps its KV in step with the target from that alone. Tokens the
target skips because it has their KV cached travel as a `prefix` with the
session's first step, so a session may start anywhere and needs no other
state on the wire.

Channels, per client:
  control   ZMQ REQ -> the engine's REP socket: Attach / Detach / Query.
  requests  ZMQ PUSH -> the engine's PULL socket: DraftRequest / Release, in
            the order the client sent them.
  tensors   a two-rank NCCL communicator: the payload of each DraftRequest
            one way, its reply the other, in request order. The engine is
            rank 0 and hosts the rendezvous.

Payload, int32, see `pack_step`: the step's GPU-resident state. It is followed
by a [num_tokens, num_hidden] tensor of the target's dtype when the drafter
needs hidden states. Reply, int32 [num_sessions, 2 * k]: k draft tokens per
session, then the float32 bits of their k acceptance confidences.
"""

from dataclasses import dataclass, field

import torch

PROTOCOL_VERSION = 3
ENGINE_RANK = 0
CLIENT_RANK = 1

METHODS = ("draft_model", "eagle3", "dflash")
# Attach.dtype: the dtype of the hidden states a client sends.
HIDDEN_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# ---------------------------------------------------------------- control


@dataclass
class DrafterInfo:
    """What an engine serves: all a target has to know about its drafter."""

    method: str
    # The drafter checkpoint.
    model: str
    # The target a head was trained for; "" for a draft model, which drafts
    # for any target with its vocabulary.
    target_model: str
    num_speculative_tokens: int
    # Layers of the target whose hidden states the drafter reads for every
    # token of a step. None: it reads none. (): the target model's default
    # auxiliary layers.
    hidden_layers: tuple[int, ...] | None
    # Whether it drafts for steps sized by adaptive verification. An engine
    # that cannot estimate confidences answers with ones.
    adaptive_verification: bool


@dataclass
class Attach:
    """Client -> engine: the target and the steps it will send. Answered
    with `Attached`."""

    target_model: str
    num_speculative_tokens: int
    # The hidden states the client sends per token: the target's layers, in
    # this order, each `hidden_size` wide (filled in by DrafterClient).
    hidden_size: int
    dtype: str
    # The widest step the client can send: its max_num_seqs sessions and
    # its max_num_batched_tokens tokens.
    max_num_seqs: int
    max_num_batched_tokens: int
    adaptive_verification: bool = False
    hidden_layers: tuple[int, ...] = ()
    label: str = ""
    version: int = PROTOCOL_VERSION

    @property
    def num_hidden(self) -> int:
        """Width of the hidden states per token (0: none)."""
        return len(self.hidden_layers) * self.hidden_size


@dataclass
class Attached:
    """`client_id` < 0 refuses, with `error`."""

    client_id: int
    request_port: int = 0
    nccl_port: int = 0
    error: str = ""


@dataclass
class Detach:
    """Client -> engine: drop the client and its sessions."""

    client_id: int


@dataclass
class Query:
    """Anyone -> engine. Answered with `Status`. From a client with its own
    id it is also a heartbeat: once a client has sent one, the engine takes
    its silence for its death."""

    client_id: int = -1


@dataclass
class SetDraftTokens:
    """Anyone -> engine: draft at most this many tokens per session from the
    next round on (cut to 0..k). Answered with `Status`."""

    num_tokens: int


@dataclass
class ClientStatus:
    client_id: int
    label: str
    # Sessions the engine holds KV for.
    num_sessions: int
    # Whether a step of the client is waiting for a round.
    waiting: bool
    # Steps drafted, and the total time their payloads waited for a round.
    num_steps: int
    queue_time_s: float


@dataclass
class Counters:
    """What the engine has done since it started; take the difference of two
    queries for a rate.

    A step waits `queue` from the landing of its payload to the launch of its
    round, because the GPU was busy or the round full, and then for the round
    itself. `round_time_s` over the elapsed time is how busy the GPU is.
    """

    # Requests per round > 1 means steps of several clients shared rounds.
    num_rounds: int = 0
    num_round_requests: int = 0
    num_round_sessions: int = 0
    # Draft tokens per session a round ran for, summed over the rounds.
    num_round_draft_tokens: int = 0
    # Rounds launched ahead of their payload (see DrafterEngine.run).
    num_prelaunched_rounds: int = 0
    num_prefix_tokens: int = 0
    num_steps: int = 0
    queue_time_s: float = 0.0
    max_queue_time_s: float = 0.0
    round_time_s: float = 0.0
    max_round_time_s: float = 0.0
    # CPU time the loop spent launching rounds.
    launch_time_s: float = 0.0
    # Longest the engine's loop took over one pass (it serves every client
    # from one thread); the timings above cannot see the loop itself stall.
    max_loop_time_s: float = 0.0


@dataclass
class Status:
    drafter: DrafterInfo
    # Draft tokens per session, at most, right now (SetDraftTokens).
    draft_tokens: int
    scheduling_policy: str
    # A round: sessions and tokens.
    max_sessions: int
    max_num_batched_tokens: int
    # KV state: sessions the engine can hold at a time over all clients and
    # holds now, KV blocks (of `block_size` tokens), tokens in one session.
    max_live_sessions: int
    num_live_sessions: int
    num_kv_blocks: int
    free_kv_blocks: int
    block_size: int
    max_model_len: int
    # Sessions that found no row or no KV blocks and got unusable drafts,
    # since the engine started. It should stay 0.
    num_overflowed_sessions: int
    clients: list[ClientStatus]
    uptime_s: float
    counters: Counters


# --------------------------------------------------------------- requests


@dataclass
class StepLayout:
    """A step as the client's CPU knows it; the exact layout is in the
    payload. With adaptive verification the GPU sizes the sessions: the CPU
    knows how many tokens a session may have at most (`num_scheduled_tokens`)
    and, unless it has read the sizes back (`exact_starts`), a
    `query_start_loc` that splits the step's tokens evenly."""

    num_tokens: int
    # [num_sessions + 1] token offsets of the sessions in the step.
    query_start_loc: list[int]
    num_scheduled_tokens: list[int]
    # Per session: an upper bound of its length after the step (the CPU does
    # not know how many drafts were rejected), and whether the step is a
    # prompt chunk for it.
    seq_lens_upper_bound: list[int]
    is_prefilling: list[bool]
    # InputBatch.decode_graph_eligible and InputBatch.max_query_len.
    decode_graph_eligible: bool = False
    max_query_len: int | None = None
    # Whether `query_start_loc` is the layout in the payload. A step can
    # only be cut by session (DrafterRouter) when it is.
    exact_starts: bool = True


@dataclass
class DraftRequest:
    """One target step. The payload follows over NCCL and the engine answers
    it with the drafts. A step is one unit: it is drafted in one round."""

    sessions: list[int]
    layout: StepLayout
    # Under the "priority" policy, lower goes first.
    priority: int = 0
    # session -> the tokens before its first step, for a session that does
    # not start at position 0: the target already holds KV for them (a prefix
    # cache hit) and never replays them. A draft model computes its own KV
    # for them before the step; a head cannot, it needs the hidden states.
    prefixes: dict[int, list[int]] = field(default_factory=dict)
    # Per session: how many tokens to draft for it, at most. Empty: as many
    # as the engine drafts.
    num_draft_tokens: list[int] = field(default_factory=list)
    # Filled in by DrafterClient.
    client_id: int = -1


@dataclass
class Release:
    """Sessions whose KV the engine may drop. Applied in the order of the
    request channel, so a session id may be reused by a later DraftRequest."""

    sessions: list[int]
    client_id: int = -1


# ---------------------------------------------------------------- tensors

# Per-session int32 fields of the payload, after the per-token ones.
STEP_FIELDS = (
    "seq_lens",
    "num_sampled",
    "num_rejected",
    "last_sampled",
    "next_prefill",
)


def packed_step_numel(num_sessions: int, num_tokens: int) -> int:
    return 2 * num_tokens + (num_sessions + 1) + len(STEP_FIELDS) * num_sessions


def pack_step(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    *fields: torch.Tensor,
) -> torch.Tensor:
    """[2 * num_tokens + num_sessions + 1 + 5 * num_sessions] int32."""
    assert len(fields) == len(STEP_FIELDS)
    parts = [input_ids, positions, query_start_loc, *fields]
    return torch.cat([p.reshape(-1).to(torch.int32) for p in parts])


def unpack_step(
    buf: torch.Tensor, num_sessions: int, num_tokens: int
) -> dict[str, torch.Tensor]:
    sizes = [num_tokens, num_tokens, num_sessions + 1]
    sizes += [num_sessions] * len(STEP_FIELDS)
    names = ["input_ids", "positions", "query_start_loc", *STEP_FIELDS]
    return dict(zip(names, torch.split(buf, sizes)))


def pack_reply(drafts: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
    """[num_sessions, 2 * k] int32 from int drafts and float32 confidences."""
    return torch.cat(
        [drafts.to(torch.int32), confidences.to(torch.float32).view(torch.int32)],
        dim=1,
    )


def unpack_reply(buf: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    return buf[:, :k], buf[:, k:].view(torch.float32)
