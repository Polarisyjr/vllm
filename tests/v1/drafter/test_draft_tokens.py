# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.drafter.protocol import StepLayout
from vllm.v1.drafter.runner import DraftRunner, StepInput, clip_verdict, pad_drafts
from vllm.v1.worker.gpu.spec_decode.remote.speculator import RemoteSpeculator

K = 3


def ints(*values):
    return torch.tensor(values, dtype=torch.int32)


@pytest.mark.parametrize("width", [K, 1])
def test_padding_has_confidence_zero(width):
    # A round of one draft step may return one column.
    drafts = torch.tensor([[11, 12, 13], [21, 22, 23]])[:, :width]
    draft_tokens = ints(1, 0) if width == 1 else ints(1, 3)

    out, conf = pad_drafts(
        drafts, torch.full((2, K), 0.5), draft_tokens, torch.arange(K)
    )

    last = [0, 0, 0] if width == 1 else [21, 22, 23]
    assert out.tolist() == [[11, 0, 0], last]
    assert (conf > 0).sum(dim=1).tolist() == draft_tokens.tolist()


def test_padding_is_not_graded():
    # Sessions with 2 drafted tokens whose target verified 3 (one padding).
    drafted = ints(2, 2, 2, 2)
    # Accepted: 0, 2 (the padding rejected), 3 (the padding accepted, by luck),
    # and a prompt chunk that samples nothing.
    sampled, rejected = clip_verdict(ints(1, 3, 4, 0), ints(3, 1, 0, 0), drafted)
    assert sampled.tolist() == [1, 3, 3, 0]
    assert rejected.tolist() == [2, 0, 0, 0]
    # A verdict within the drafted tokens is left alone.
    sampled, rejected = clip_verdict(ints(2), ints(1), ints(2))
    assert (sampled.tolist(), rejected.tolist()) == ([2], [1])


def test_round_takes_the_lower_of_request_and_engine():
    def step(b, num_draft_tokens):
        layout = StepLayout(
            num_tokens=b,
            query_start_loc=list(range(b + 1)),
            num_scheduled_tokens=[1] * b,
            seq_lens_upper_bound=[8] * b,
            is_prefilling=[False] * b,
        )
        return StepInput(list(range(b)), layout, None, None, num_draft_tokens)

    steps = [step(3, [0, 1, 9]), step(2, [])]
    merge = DraftRunner._merge
    assert merge(None, steps, 3).draft_tokens.tolist() == [0, 1, 3, 3, 3]
    assert merge(None, steps, 2).draft_tokens.tolist() == [0, 1, 2, 2, 2]


def test_request_names_its_draft_tokens():
    spec = SimpleNamespace(
        drafters=SimpleNamespace(
            opened=[], open=lambda *a: spec.drafters.opened.append(a)
        ),
        _draft_tokens={},
        _prefixes={},
    )

    def add(slot, **extra_args):
        params = SamplingParams(extra_args=extra_args or None)
        RemoteSpeculator.on_request_added(spec, slot, [1, 2], 0, params)

    add(0, drafter="a:7700", draft_tokens=2)
    add(1, draft_tokens="1")
    add(2)
    add(3, draft_tokens="many")
    assert spec.drafters.opened == [(0, "a:7700"), (1, None), (2, None), (3, None)]
    assert spec._draft_tokens == {0: 2, 1: 1}
    # The slot's next request asks for nothing.
    add(0)
    assert spec._draft_tokens == {1: 1}
