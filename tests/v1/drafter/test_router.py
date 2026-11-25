# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.drafter.protocol import (
    STEP_FIELDS,
    DraftRequest,
    StepLayout,
    pack_step,
    unpack_step,
)
from vllm.v1.drafter.router import DrafterRouter, split_layout, split_step

K = 2


def make_router(num_engines):
    return DrafterRouter(
        [f"host:{7700 + 10 * i}" for i in range(num_engines)], torch.device("cpu")
    )


class FakeClient:
    """An engine that drafts its sessions' ids."""

    attached = True

    def __init__(self, hidden_layers=()):
        self.hidden_layers = hidden_layers
        self.steps = []

    def submit(self, request, step, hidden=None):
        self.steps.append((request, step, hidden))
        return request

    def collect(self, pending):
        return torch.tensor([[s] * (2 * K) for s in pending.sessions]).int()

    def release(self, sessions):
        pass


def make_step(query_start_loc):
    b, n = len(query_start_loc) - 1, query_start_loc[-1]
    step = pack_step(
        torch.arange(100, 100 + n),
        torch.arange(n),
        torch.tensor(query_start_loc),
        *(torch.arange(b) for _ in STEP_FIELDS),
    )
    layout = StepLayout(
        num_tokens=n,
        query_start_loc=query_start_loc,
        num_scheduled_tokens=[4] * b,
        seq_lens_upper_bound=[40] * b,
        is_prefilling=[False] * b,
    )
    return step, layout


def test_sessions_spread_and_stay():
    router = make_router(2)
    assert router._route([10, 11, 12, 13]) == {0: [0, 2], 1: [1, 3]}
    # Same sessions, another order in the step: same engines.
    assert router._route([13, 12, 11, 10]) == {1: [0, 2], 0: [1, 3]}

    router.release([10, 12])
    # Engine 0 is empty now, so it gets the next two.
    assert router._route([11, 13, 20, 21]) == {1: [0, 1], 0: [2, 3]}


def test_one_engine_gets_everything():
    router = make_router(1)
    assert router._route([3, 1, 2]) == {0: [0, 1, 2]}


def test_session_goes_to_the_engine_it_was_paired_with():
    router = make_router(2)
    router.open(1, "host:7710")
    router.open(2, "host:7710")
    # No pairing, or an engine this target does not have: by load.
    router.open(3)
    router.open(4, "elsewhere:7700")
    assert router._route([1, 2, 3, 4]) == {1: [0, 1], 0: [2, 3]}

    # A slot's next request may be paired with another engine.
    router.open(1, "host:7700")
    assert router._route([1, 2]) == {0: [0], 1: [1]}


def test_step_is_cut_for_engines_of_different_kinds():
    router = make_router(2)
    draft_model, head = FakeClient(), FakeClient(hidden_layers=(2, 5))
    router.clients = [draft_model, head]
    router._k = K
    for session, endpoint in ((10, "host:7710"), (11, "host:7700"), (12, "host:7710")):
        router.open(session, endpoint)
    step, layout = make_step([0, 3, 4, 6])
    # The target exports a layer more than this head reads.
    hidden = {layer: torch.arange(6.0)[:, None] + 100 * layer for layer in (2, 5, 9)}

    pending = router.submit(DraftRequest([10, 11, 12], layout), step, hidden)

    ((request, _, sent),) = draft_model.steps
    assert request.sessions == [11] and sent is None
    assert request.layout.query_start_loc == [0, 1]
    ((request, sub_step, sent),) = head.steps
    assert request.sessions == [10, 12]
    assert unpack_step(sub_step, 2, 5)["input_ids"].tolist() == [
        100,
        101,
        102,
        104,
        105,
    ]
    assert sorted(sent) == [2, 5]
    assert sent[5][:, 0].tolist() == [500, 501, 502, 504, 505]
    # The replies come back in the order of the step.
    assert router.collect(pending)[:, 0].tolist() == [10, 11, 12]


def test_step_sized_on_the_gpu_is_cut_only_with_its_exact_layout():
    router = make_router(2)
    router.clients = [FakeClient(), FakeClient()]
    step, layout = make_step([0, 3, 5, 7])
    # Adaptive verification: the CPU has an even split.
    layout.max_query_len, layout.exact_starts = 4, False

    with pytest.raises(ValueError, match="exact"):
        router.submit(DraftRequest([1, 2, 3], layout), step)

    # One engine gets the whole step: nothing to cut.
    router = make_router(1)
    router.clients = [FakeClient()]
    router.submit(DraftRequest([1, 2, 3], layout), step)


def test_split_layout():
    layout = StepLayout(
        num_tokens=9,
        query_start_loc=[0, 4, 5, 9],
        num_scheduled_tokens=[4, 1, 4],
        seq_lens_upper_bound=[40, 50, 60],
        is_prefilling=[False, True, False],
        decode_graph_eligible=False,
    )

    sub, tokens = split_layout(layout, [0, 2])

    assert tokens.tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    assert sub.num_tokens == 8
    assert sub.query_start_loc == [0, 4, 8]
    assert sub.num_scheduled_tokens == [4, 4]
    assert sub.seq_lens_upper_bound == [40, 60]
    assert sub.is_prefilling == [False, False]


def test_split_layout_with_sizes_from_the_gpu():
    # Adaptive verification: the sessions take what the GPU gave them, up to
    # num_scheduled_tokens each.
    layout = StepLayout(
        num_tokens=7,
        query_start_loc=[0, 4, 5, 7],
        num_scheduled_tokens=[4, 4, 4],
        seq_lens_upper_bound=[40, 50, 60],
        is_prefilling=[False] * 3,
        max_query_len=4,
    )

    sub, tokens = split_layout(layout, [1, 2])

    assert tokens.tolist() == [4, 5, 6]
    assert sub.query_start_loc == [0, 1, 3]
    assert sub.num_scheduled_tokens == [4, 4] and sub.max_query_len == 4


def test_split_step_round_trip():
    b, n = 3, 9
    query_start_loc = [0, 4, 5, 9]
    fields = [torch.arange(b) + 10 * (i + 1) for i in range(len(STEP_FIELDS))]
    step = pack_step(
        torch.arange(100, 100 + n),
        torch.arange(n),
        torch.tensor(query_start_loc),
        *fields,
    )
    layout = StepLayout(
        num_tokens=n,
        query_start_loc=query_start_loc,
        num_scheduled_tokens=[4, 1, 4],
        seq_lens_upper_bound=[40, 50, 60],
        is_prefilling=[False] * 3,
    )
    positions = [2, 0]

    sub, tokens = split_layout(layout, positions)
    sub_step = split_step(
        unpack_step(step, b, n),
        torch.tensor(positions),
        torch.from_numpy(tokens),
        torch.tensor(sub.query_start_loc, dtype=torch.int32),
    )

    out = unpack_step(sub_step, len(positions), sub.num_tokens)
    assert out["input_ids"].tolist() == [105, 106, 107, 108, 100, 101, 102, 103]
    assert out["positions"].tolist() == [5, 6, 7, 8, 0, 1, 2, 3]
    assert out["query_start_loc"].tolist() == [0, 4, 8]
    for name, value in zip(STEP_FIELDS, fields):
        assert out[name].tolist() == value[positions].tolist()
