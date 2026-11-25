# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.drafter.links import Client
from vllm.v1.drafter.protocol import StepLayout
from vllm.v1.drafter.runner import StepInput
from vllm.v1.drafter.scheduler import Scheduler, Step


class _Event:
    def __init__(self, landed: bool):
        self.landed = landed

    def query(self) -> bool:
        return self.landed


def make_step(client_id, num_sessions, num_tokens, priority=0, landed=True):
    layout = StepLayout(
        num_tokens=num_tokens,
        query_start_loc=[],
        num_scheduled_tokens=[],
        seq_lens_upper_bound=[],
        is_prefilling=[],
    )
    return Step(
        client=Client(id=client_id, attach=None),
        priority=priority,
        input=StepInput(rows=list(range(num_sessions)), layout=layout, step=None),
        landed=_Event(landed),
        # Clients arrive in the order of their ids.
        arrived=float(client_id),
    )


def make_scheduler(policy="fcfs", max_sessions=64, max_num_tokens=1024, **kwargs):
    return Scheduler(policy, max_sessions, max_num_tokens, 3, **kwargs)


def scheduled(scheduler, gpu_free=True):
    return [step.client.id for step in scheduler.schedule(gpu_free)]


def test_fits():
    scheduler = make_scheduler(max_sessions=64, max_num_tokens=1024)
    assert scheduler.fits(64, 960)
    assert not scheduler.fits(65, 65)
    # A draft model adds a slot per session to its first pass.
    assert not scheduler.fits(64, 961)
    # DFlash runs k + 1 query tokens per session.
    assert not make_scheduler(max_sessions=512).fits(257, 257)


def test_round_waits_for_gpu_and_payload():
    scheduler = make_scheduler()
    step = make_step(0, 8, 8, landed=False)
    scheduler.add(step)

    assert scheduled(scheduler) == []
    step.landed.landed = True
    assert scheduled(scheduler, gpu_free=False) == []
    # Queue time counts from the landing, not from the GPU getting free.
    assert step.landed_at is not None
    assert scheduled(scheduler) == [0]
    assert len(scheduler) == 0


def test_steps_share_a_round():
    scheduler = make_scheduler()
    for client_id in range(3):
        scheduler.add(make_step(client_id, 16, 16, landed=client_id != 1))

    assert scheduled(scheduler) == [0, 2]
    assert 1 in scheduler


def test_step_that_does_not_fit_waits():
    scheduler = make_scheduler(max_sessions=64)
    scheduler.add(make_step(0, 48, 48))
    scheduler.add(make_step(1, 32, 32))
    scheduler.add(make_step(2, 16, 16))

    # Oldest first; a later step may take the room an earlier one cannot.
    assert scheduled(scheduler) == [0, 2]
    assert scheduled(scheduler) == [1]


def test_priority_policy():
    for policy, order in (("fcfs", [0]), ("priority", [1])):
        scheduler = make_scheduler(policy, max_sessions=64)
        scheduler.add(make_step(0, 64, 64, priority=1))
        scheduler.add(make_step(1, 64, 64, priority=0))
        assert scheduled(scheduler) == order


def test_batch_window_waits_for_payloads_in_flight():
    scheduler = make_scheduler(batch_window=3600.0)
    scheduler.add(make_step(0, 8, 8))
    late = make_step(1, 8, 8, landed=False)
    scheduler.add(late)

    assert scheduled(scheduler) == []
    late.landed.landed = True
    assert scheduled(scheduler) == [0, 1]


def test_one_step_per_client():
    scheduler = make_scheduler()
    scheduler.add(make_step(0, 8, 8))
    with pytest.raises(AssertionError):
        scheduler.add(make_step(0, 8, 8))
