# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft sessions for a drafter service.

A drafter service runs an ordinary engine whose requests include *draft
sessions*: one per live request of a remote target engine. A session is a
resumable (streaming) request that drafts k tokens per round and then parks
as WAITING_FOR_STREAMING_REQ, keeping its KV blocks and its model-runner
slot. The target's verify result arrives as the next round:

    rollback_to   the target's verified length (tokens the target holds KV
                  for); the session is truncated to it, which may drop
                  drafts the target rejected (below the computed frontier)
                  or keep a sampled-but-uncomputed draft the target accepted
    prompt        the bonus token the target sampled after that prefix
    max_tokens    k, the next round's draft length

The stock streaming update keeps every computed token; the only change here
is the truncation, applied before the update. The scheduler then recomputes
the uncomputed tail (at most one accepted draft plus the bonus) through the
chunked-prefill path and decodes the rest of the round.

Select with ``scheduler_cls="vllm.v1.core.sched.draft_session.
DraftSessionAsyncScheduler"`` (or ``DraftSessionScheduler`` without async
scheduling). Prefix caching must be off: a rollback rewrites KV in place,
which would invalidate blocks cached by hash.
"""

from dataclasses import dataclass

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus, StreamingUpdate


@dataclass
class DraftRoundUpdate(StreamingUpdate):
    rollback_to: int = -1

    @classmethod
    def from_request(cls, request: Request) -> "DraftRoundUpdate | None":
        base = StreamingUpdate.from_request(request)
        if base is None:
            return None
        return cls(**base.__dict__, rollback_to=getattr(request, "rollback_to", -1))


class DraftSessionMixin:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.cache_config.enable_prefix_caching:
            raise ValueError("draft sessions require enable_prefix_caching=False")

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is None or getattr(request, "rollback_to", -1) < 0:
            return super().add_request(request)
        update = DraftRoundUpdate.from_request(request)
        assert update is not None
        if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
            # The previous round is still running; applied when it parks.
            assert existing.streaming_queue is not None, "duplicate request id"
            existing.streaming_queue.append(update)
        else:
            self._update_request_as_session(existing, update)

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        rollback_to = getattr(update, "rollback_to", -1)
        if rollback_to < 0:
            return super()._update_request_as_session(session, update)
        assert rollback_to <= session.num_tokens, (rollback_to, session.num_tokens)
        # Truncate to the target's verified prefix. Tokens beyond the computed
        # frontier (an accepted draft that was sampled but not yet computed)
        # stay and are recomputed with the bonus token.
        session.num_computed_tokens = min(session.num_computed_tokens, rollback_to)
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : rollback_to
        ]
        del session._all_token_ids[rollback_to:]
        del session.block_hashes[rollback_to // self.block_size :]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        session.prompt_token_ids.extend(kept_output_tokens)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        session.max_tokens = update.max_tokens
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING


class DraftSessionAsyncScheduler(DraftSessionMixin, AsyncScheduler):
    pass


class DraftSessionScheduler(DraftSessionMixin, Scheduler):
    pass
