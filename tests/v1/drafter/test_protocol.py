# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.v1.drafter.protocol import (
    STEP_FIELDS,
    pack_reply,
    pack_step,
    packed_step_numel,
    unpack_reply,
    unpack_step,
)


def test_step_round_trip():
    b, n = 3, 7
    input_ids = torch.arange(100, 100 + n)
    positions = torch.arange(n)
    query_start_loc = torch.tensor([0, 2, 5, 7])
    fields = [torch.arange(b) + 10 * i for i in range(len(STEP_FIELDS))]

    buf = pack_step(input_ids, positions, query_start_loc, *fields)

    assert buf.dtype == torch.int32
    assert buf.numel() == packed_step_numel(b, n)
    out = unpack_step(buf, b, n)
    assert out["input_ids"].tolist() == input_ids.tolist()
    assert out["positions"].tolist() == positions.tolist()
    assert out["query_start_loc"].tolist() == query_start_loc.tolist()
    for name, value in zip(STEP_FIELDS, fields):
        assert out[name].tolist() == value.tolist()


def test_reply_round_trip():
    k = 3
    drafts = torch.tensor([[1, 2, 3], [4, 5, 6]])
    confidences = torch.tensor([[0.9, 0.5, 0.25], [1.0, 0.0, 0.125]])

    buf = pack_reply(drafts, confidences)

    assert buf.dtype == torch.int32 and buf.shape == (2, 2 * k)
    out_drafts, out_confidences = unpack_reply(buf, k)
    assert out_drafts.tolist() == drafts.tolist()
    assert torch.equal(out_confidences, confidences)
