# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire protocol between a target engine and a drafter service.

The target keeps one *session* per live request at the drafter, keyed by the
request's slot index in the target's RequestState (stable while the request is
live, reused afterwards). Messages on the step socket are ordered, so a slot
may be released and re-opened in the same header: `released` is applied
before `new`.

Per step the target ships, for every row of its batch, three int32 values
that live on the GPU (async scheduling never materialises them on the CPU):

    num_computed  tokens the target holds KV for after this step's verify
    num_sampled   accepted drafts + 1 (bonus), 0 for a prefill chunk that
                  did not finish
    bonus         the token sampled last (valid when num_sampled > 0)

The verified sequence of a row is its first `num_computed` tokens followed by
`bonus`; the drafter rolls its session back to `num_computed`, appends the
bonus and drafts k tokens from there. Rows with num_sampled == 0 need no
drafts. Hidden-state heads (EAGLE-3, DFlash) add the packed target step and
the hidden states (payload "hidden"); that part of the protocol is reserved
here and filled in by the head service.
"""

from dataclasses import dataclass, field

import torch

TARGET_RANK = 0
DRAFTER_RANK = 1
TOKENS_PER_ROW = 3  # num_computed, num_sampled, bonus


@dataclass
class Hello:
    """Target -> service, once, on the control socket."""

    method: str  # draft_model | eagle3 | dflash
    target_model: str
    draft_model: str
    dtype: str
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    num_speculative_tokens: int
    # Width of the hidden states a head needs (0 for a token-only drafter);
    # EAGLE-3 ships num_aux layers of them.
    hidden_size: int = 0
    num_aux_hidden: int = 0
    target_label: str = ""


@dataclass
class Ready:
    """Service -> target, reply to Hello."""

    target_id: int
    step_port: int
    nccl_port: int
    payload: str = "tokens"  # tokens | hidden


@dataclass
class NewSession:
    slot: int
    # Prompt tokens, plus the replayed outputs when a preempted request is
    # resumed.
    token_ids: list[int]


@dataclass
class StepHeader:
    """One message per target step on the step socket (pickled over ZMQ).

    With `has_step`, an int32 tensor of TOKENS_PER_ROW * num_reqs values
    follows over NCCL and the service answers with int64 drafts
    [num_reqs, k]. Without it the header only carries session control.
    """

    target_id: int
    step: int
    has_step: bool
    slots: list[int] = field(default_factory=list)
    k: int = 0
    released: list[int] | None = None
    new: list[NewSession] | None = None
    # Hidden payload (heads): the packed target step (see pack_step) is an
    # int32 tensor followed by a [num_tokens, num_hidden] tensor of the
    # target's dtype; the CPU-side batch layout travels in the header.
    num_tokens: int = 0
    num_hidden: int = 0
    query_start_loc: list[int] | None = None
    # Per row: tokens computed before the step (upper bound) and whether the
    # row is a prefill chunk (then the head's prefill graph is not uniform).
    num_computed: list[int] | None = None
    is_prefilling: list[bool] | None = None


def pack_rows(
    num_computed: torch.Tensor, num_sampled: torch.Tensor, bonus: torch.Tensor
) -> torch.Tensor:
    """[3 * num_reqs] int32: the three per-row values, row-major by field."""
    return torch.stack(
        [num_computed.to(torch.int32), num_sampled.to(torch.int32), bonus.to(torch.int32)]
    ).reshape(-1)


def unpack_rows(buf: torch.Tensor, num_reqs: int) -> tuple[torch.Tensor, ...]:
    return tuple(buf.view(TOKENS_PER_ROW, num_reqs))


STEP_FIELDS = ("num_sampled", "num_rejected", "last_sampled", "next_prefill")


def packed_step_numel(num_reqs: int, num_tokens: int) -> int:
    return 2 * num_tokens + (num_reqs + 1) + num_reqs * (1 + len(STEP_FIELDS))


def pack_step(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill: torch.Tensor,
) -> torch.Tensor:
    """Hidden-payload step: the target batch's token layout plus the verify
    result, as one int32 tensor (all GPU-resident on the target)."""
    parts = [input_ids, positions, query_start_loc, seq_lens, num_sampled,
             num_rejected, last_sampled, next_prefill]
    return torch.cat([p.reshape(-1).to(torch.int32) for p in parts])


def unpack_step(buf: torch.Tensor, num_reqs: int, num_tokens: int) -> dict[str, torch.Tensor]:
    b, n = num_reqs, num_tokens
    sizes = [n, n, b + 1, b, b, b, b, b]
    names = ["input_ids", "positions", "query_start_loc", "seq_lens", *STEP_FIELDS]
    return dict(zip(names, torch.split(buf, sizes)))
