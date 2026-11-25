# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafter engine: the loop that turns clients' steps into rounds.

The engine is request driven. A client attaches whenever it likes and sends a
request per target step. One pass of the loop:

  1. control    answer Attach / Detach / Query;
  2. receive    move requests from the socket to their clients' inboxes;
  3. take       per client, the next request: give its sessions rows and KV
                blocks, start receiving its payload on the client's own CUDA
                stream (a client whose target is still computing never holds
                up another one), and put the step in the pending pool;
  4. schedule   once the GPU has finished the last round, the scheduler picks
                the landed steps of the next one;
  5. round      one `propose()` over the sessions of all of them, and one
                reply per step, sent on its client's stream once the GPU
                has the drafts.

Nothing in the loop waits for the GPU, and nothing a client sends or
receives is on the engine's own stream, except a round launched ahead of its
payload: a client that goes away holds up only itself.
"""

import json
import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch
import zmq

from vllm.logger import init_logger
from vllm.v1.drafter.config import DrafterConfig
from vllm.v1.drafter.links import Client, link, listen
from vllm.v1.drafter.protocol import (
    CLIENT_RANK,
    HIDDEN_DTYPES,
    PROTOCOL_VERSION,
    Attach,
    Attached,
    ClientStatus,
    Counters,
    Detach,
    DrafterInfo,
    DraftRequest,
    Query,
    Release,
    SetDraftTokens,
    Status,
    pack_reply,
    packed_step_numel,
)
from vllm.v1.drafter.runner import DraftRunner, StepInput
from vllm.v1.drafter.scheduler import Scheduler, Step

logger = init_logger(__name__)


@dataclass
class _Round:
    """A round the GPU has not finished."""

    done: torch.cuda.Event
    # When the GPU could start on it: its launch, or for a round launched
    # ahead of its payload, the payload's landing.
    started: float
    # Payload events of a round launched ahead, until they are seen landed.
    landing: list[torch.cuda.Event] = field(default_factory=list)


class DrafterEngine:
    def __init__(self, config: DrafterConfig):
        self.config = config
        self.runner = DraftRunner(config)
        self.device = self.runner.device
        self.scheduler = Scheduler(
            config.scheduling_policy,
            config.max_sessions,
            self.runner.max_num_tokens,
            self.runner.k,
            batch_window=config.batch_window_ms / 1e3,
        )
        # A batch window holds landed steps back, a prelaunch runs ahead.
        self.prelaunch = config.prelaunch and config.batch_window_ms == 0
        # Draft tokens per session, at most (SetDraftTokens).
        self.draft_tokens = self.runner.k
        self.clients: dict[int, Client] = {}
        self._next_client_id = 0
        # Detached clients whose link still has a transfer in flight.
        self._unlinking: list[Client] = []
        self._main = torch.cuda.current_stream(self.device)
        # Rounds the GPU has not finished, oldest first.
        self._rounds: deque[_Round] = deque()
        self._last_round_done = 0.0
        self.counters = Counters()
        self._started = time.perf_counter()
        self._stopping = False

        self.host = config.host or socket.gethostname()
        self.request_port = config.port + 1
        self._zmq = zmq.Context()
        self._control = self._zmq.socket(zmq.REP)
        self._control.bind(f"tcp://*:{config.port}")
        self._requests = self._zmq.socket(zmq.PULL)
        self._requests.bind(f"tcp://*:{self.request_port}")
        self._poller = zmq.Poller()
        self._poller.register(self._control, zmq.POLLIN)
        self._poller.register(self._requests, zmq.POLLIN)
        logger.info("Drafter engine listening on %s:%d", self.host, config.port)

    # ----------------------------------------------------------------- loop

    def run(self) -> None:
        interval = self.config.stats_interval
        next_stats = time.perf_counter() + interval
        next_reap = time.perf_counter() + 1.0
        polled = time.perf_counter()
        try:
            while not self._stopping:
                busy = (
                    bool(self._rounds)
                    or len(self.scheduler) > 0
                    or any(c.linked and c.inbox for c in self.clients.values())
                )
                now = time.perf_counter()
                self.counters.max_loop_time_s = max(
                    self.counters.max_loop_time_s, now - polled
                )
                socks = dict(self._poller.poll(0 if busy else 20))
                polled = now = time.perf_counter()
                if self._control in socks:
                    self._on_control()
                self._receive()
                if now > next_reap:
                    next_reap = now + 1.0
                    self._reap()
                if interval > 0 and now > next_stats:
                    next_stats = now + interval
                    self._log_stats()

                taken = self._take()
                # A target sends a request a step ahead of its payload. If
                # the step is alone in the engine, its round is enqueued
                # right away, behind the payload on the stream, and the GPU
                # starts drafting the moment the payload lands, as it does
                # for a drafter inside the target. With other steps pending
                # this could put a landed payload behind one that has not,
                # so those rounds wait.
                if self.prelaunch and len(taken) == 1 and len(self.scheduler) == 1:
                    self.scheduler.pop(taken[0].client.id)
                    self.counters.num_prelaunched_rounds += 1
                    self._round(taken)
                    continue
                steps = self.scheduler.schedule(self._gpu_free())
                if steps:
                    self._round(steps)
        finally:
            self._log_stats()

    def stop(self) -> None:
        self._stopping = True

    # -------------------------------------------------------------- clients

    def _on_control(self) -> None:
        msg = self._control.recv_pyobj()
        if isinstance(msg, Query):
            client = self.clients.get(msg.client_id)
            if client is not None:
                client.last_seen = time.perf_counter()
                client.heartbeats = True
            self._control.send_pyobj(self.status())
        elif isinstance(msg, SetDraftTokens):
            self.draft_tokens = max(0, min(msg.num_tokens, self.runner.k))
            logger.info("Drafting up to %d tokens per session", self.draft_tokens)
            self._control.send_pyobj(self.status())
        elif isinstance(msg, Detach):
            self._control.send_pyobj(True)
            self._detach(msg.client_id)
        else:
            assert isinstance(msg, Attach)
            self._control.send_pyobj(self._attach(msg))

    def _attach(self, attach: Attach) -> Attached:
        error = self._refusal(attach)
        if error:
            logger.warning("Refused client %s: %s", attach.label, error)
            return Attached(client_id=-1, error=error)
        client = Client(id=self._next_client_id, attach=attach)
        self._next_client_id += 1
        # The client's requests are kept in its inbox until it is linked.
        self.clients[client.id] = client
        listener, nccl_port = listen(self.host)
        threading.Thread(
            target=link,
            args=(client, listener, self.host, nccl_port, self.device),
            name=f"drafter-link-{client.id}",
            daemon=True,
        ).start()
        logger.info(
            "Client %d attached: %s (%d clients)",
            client.id,
            attach.label,
            len(self.clients),
        )
        return Attached(
            client_id=client.id, request_port=self.request_port, nccl_port=nccl_port
        )

    def _refusal(self, a: Attach) -> str:
        """Why this engine cannot draft for the client, or ""."""
        runner, config = self.runner, self.config
        if a.version != PROTOCOL_VERSION:
            return f"protocol version {a.version}, engine speaks {PROTOCOL_VERSION}"
        info = self.info()

        def name(model: str) -> str:
            return os.path.basename(model.rstrip("/"))

        if info.target_model and name(a.target_model) != name(info.target_model):
            return (
                f"engine serves a {info.method} head for {info.target_model}, "
                f"not for {a.target_model}"
            )
        if a.num_speculative_tokens != runner.k:
            return f"engine drafts {runner.k} tokens, not {a.num_speculative_tokens}"
        if a.adaptive_verification and not config.adaptive_verification:
            return "engine was not started with adaptive verification"
        if bool(a.hidden_layers) != (info.hidden_layers is not None):
            return f"a {info.method} drafter reads hidden states, a draft model none"
        if a.hidden_layers:
            if a.dtype not in HIDDEN_DTYPES:
                return f"unsupported hidden state dtype {a.dtype}"
            if a.hidden_size != runner.target_hidden_size:
                return (
                    f"engine reads hidden states {runner.target_hidden_size} wide, "
                    f"not {a.hidden_size}"
                )
            if info.hidden_layers and tuple(a.hidden_layers) != info.hidden_layers:
                return (
                    f"engine reads the target's layers {info.hidden_layers}, "
                    f"not {tuple(a.hidden_layers)}"
                )
        # A step is drafted in one round, so the widest one must fit.
        if not self.scheduler.fits(a.max_num_seqs, a.max_num_batched_tokens):
            return (
                f"a step of {a.max_num_seqs} sessions and "
                f"{a.max_num_batched_tokens} tokens exceeds a round of this "
                f"engine ({self.scheduler.max_sessions} sessions, "
                f"{self.scheduler.max_num_tokens} tokens)"
            )
        return ""

    def _detach(self, client_id: int, dead: bool = False) -> None:
        client = self.clients.pop(client_id, None)
        if client is None:
            return
        self.scheduler.pop(client_id)
        for row in client.sessions.values():
            self.runner.sessions.close(row)
        logger.info(
            "Client %d detached after %d steps (%d clients)",
            client_id,
            client.num_steps,
            len(self.clients),
        )
        if client.comm is None:
            return
        if dead:
            # Its transfers will never complete; aborting the link is the
            # only way to end them.
            threading.Thread(
                target=client.comm.destroy,
                name=f"drafter-abort-{client.id}",
                daemon=True,
            ).start()
        elif not self._unlink(client):
            self._unlinking.append(client)

    def _unlink(self, client: Client) -> bool:
        """Free the client's link (its GPU buffers: NCCL runs out after a few
        links otherwise) if nothing is in flight on it. Destroying a link
        with a transfer in flight can block every CUDA call of the process."""
        assert client.comm is not None and client.stream is not None
        if not client.stream.query():
            return False
        # Off the loop: it may still take a while.
        threading.Thread(
            target=client.comm.destroy, name=f"drafter-unlink-{client.id}", daemon=True
        ).start()
        return True

    def _reap(self) -> None:
        """Drop clients whose link failed, and clients that went quiet: one
        that holds sessions or owes a payload (a live target with requests in
        flight sends a step every few ms), or one that sends heartbeats."""
        now = time.perf_counter()
        self._unlinking = [c for c in self._unlinking if not self._unlink(c)]
        for client in list(self.clients.values()):
            if client.failed:
                self._detach(client.id)
                continue
            holds = client.sessions or client.id in self.scheduler
            silent = now - client.last_seen > self.config.idle_timeout
            if silent and (holds or client.heartbeats):
                logger.warning(
                    "Client %d silent for %.0f s with %d sessions: dropping it",
                    client.id,
                    now - client.last_seen,
                    len(client.sessions),
                )
                self._detach(client.id, dead=True)

    # ------------------------------------------------------------- requests

    def _receive(self) -> None:
        while self._requests.poll(0):
            msg: DraftRequest | Release = self._requests.recv_pyobj()
            client = self.clients.get(msg.client_id)
            if client is not None:
                client.inbox.append(msg)
                client.last_seen = time.perf_counter()

    def _take(self) -> list[Step]:
        """Take requests from the inboxes until every linked client has a
        step in the pool or nothing left. Returns the new steps."""
        steps = []
        for client in self.clients.values():
            while client.linked and client.inbox and client.id not in self.scheduler:
                msg = client.inbox.popleft()
                if isinstance(msg, Release):
                    self._release(client, msg.sessions)
                else:
                    steps.append(self._begin(client, msg))
        return steps

    def _release(self, client: Client, sessions: list[int]) -> None:
        for session in sessions:
            row = client.sessions.pop(session, None)
            if row is not None:
                self.runner.sessions.close(row)

    def _begin(self, client: Client, request: DraftRequest) -> Step:
        """Give the request's sessions their rows and KV blocks, start
        receiving its payload on the client's stream and pool the step."""
        store, layout = self.runner.sessions, request.layout
        rows = []
        for session, seq_len in zip(request.sessions, layout.seq_lens_upper_bound):
            row = client.sessions.get(session)
            if row is None:
                row = store.open()
            row = client.sessions[session] = store.reserve(row, seq_len)
            rows.append(row)
        if request.prefixes:
            self._prefill(client, request.prefixes)

        assert client.comm is not None and client.stream is not None
        b, n = len(rows), layout.num_tokens
        num_hidden = client.attach.num_hidden
        with torch.cuda.stream(client.stream):
            payload = torch.empty(
                packed_step_numel(b, n), dtype=torch.int32, device=self.device
            )
            client.comm.recv(payload, src=CLIENT_RANK, stream=client.stream)
            hidden = None
            if num_hidden:
                hidden = torch.empty(
                    (n, num_hidden),
                    dtype=HIDDEN_DTYPES[client.attach.dtype],
                    device=self.device,
                )
                client.comm.recv(hidden, src=CLIENT_RANK, stream=client.stream)
            landed = torch.cuda.Event()
            landed.record(client.stream)
        step = Step(
            client=client,
            priority=request.priority,
            input=StepInput(rows, layout, payload, hidden, request.num_draft_tokens),
            landed=landed,
            arrived=time.perf_counter(),
        )
        self.scheduler.add(step)
        return step

    def _prefill(self, client: Client, prefixes: dict[int, list[int]]) -> None:
        """Compute the KV of tokens the target will not replay, before the
        sessions' first step. No reply."""
        if self.runner.method != "draft_model":
            raise RuntimeError(
                f"client {client.id} starts a session past position 0 (a "
                f"prefix cache hit on the target); a {self.runner.method} head "
                "cannot rebuild its KV without the target's hidden states"
            )
        rows = [client.sessions[session] for session in prefixes]
        self.counters.num_prefix_tokens += self.runner.prefill(
            rows, list(prefixes.values())
        )
        self._track(time.perf_counter(), [])

    # --------------------------------------------------------------- rounds

    def _round(self, steps: list[Step]) -> None:
        """One propose() over the steps, one reply each."""
        t0 = time.perf_counter()
        counters = self.counters
        landing = []
        for step in steps:
            if step.landed_at is None:
                # Launched ahead of its payload: it never waits for a round.
                landing.append(step.landed)
                waited = 0.0
            else:
                waited = t0 - step.landed_at
            step.client.num_steps += 1
            step.client.queue_s += waited
            counters.num_steps += 1
            counters.queue_time_s += waited
            counters.max_queue_time_s = max(counters.max_queue_time_s, waited)
            self._main.wait_event(step.landed)
            # Allocated on the client's stream, read on this one.
            step.input.step.record_stream(self._main)
            if step.input.hidden is not None:
                step.input.hidden.record_stream(self._main)
        drafts, confidences = self.runner.propose(
            [step.input for step in steps], self.draft_tokens
        )
        replies = []
        start = 0
        for step in steps:
            end = start + step.num_sessions
            replies.append(pack_reply(drafts[start:end], confidences[start:end]))
            start = end
        drafted = self._track(t0, landing)
        # A reply leaves on its client's stream, not this one: a client that
        # does not take its reply must not hold up the next round.
        for step, reply in zip(steps, replies):
            client = step.client
            assert client.comm is not None and client.stream is not None
            client.stream.wait_event(drafted)
            reply.record_stream(client.stream)
            client.comm.send(reply, dst=CLIENT_RANK, stream=client.stream)
        counters.num_rounds += 1
        counters.num_round_requests += len(steps)
        counters.num_round_sessions += start
        counters.num_round_draft_tokens += self.runner.round_draft_tokens
        counters.launch_time_s += time.perf_counter() - t0

    def _track(
        self, started: float, landing: list[torch.cuda.Event]
    ) -> torch.cuda.Event:
        """Follow the round just enqueued; returns the event of its end."""
        done = torch.cuda.Event()
        done.record(self._main)
        self._rounds.append(_Round(done, started, landing))
        return done

    def _gpu_free(self) -> bool:
        """Whether the GPU has finished every round; times those it has."""
        counters = self.counters
        while self._rounds:
            r = self._rounds[0]
            now = time.perf_counter()
            if r.landing:
                r.landing = [e for e in r.landing if not e.query()]
                if r.landing:
                    return False
                r.started = now
            if not r.done.query():
                return False
            # A round launched while the last one ran started when that ended.
            spent = now - max(r.started, self._last_round_done)
            counters.round_time_s += spent
            counters.max_round_time_s = max(counters.max_round_time_s, spent)
            self._last_round_done = now
            self._rounds.popleft()
        return True

    # ---------------------------------------------------------------- status

    def info(self) -> DrafterInfo:
        runner, config = self.runner, self.config
        is_head = runner.hidden_layers is not None
        return DrafterInfo(
            method=runner.method,
            model=config.model,
            target_model=(config.target_model or "") if is_head else "",
            num_speculative_tokens=runner.k,
            hidden_layers=runner.hidden_layers,
            adaptive_verification=config.adaptive_verification,
        )

    def status(self) -> Status:
        runner, config, store = self.runner, self.config, self.runner.sessions
        clients = [
            ClientStatus(
                client_id=c.id,
                label=c.attach.label,
                num_sessions=len(c.sessions),
                waiting=c.id in self.scheduler or bool(c.inbox),
                num_steps=c.num_steps,
                queue_time_s=c.queue_s,
            )
            for c in self.clients.values()
        ]
        return Status(
            drafter=self.info(),
            draft_tokens=self.draft_tokens,
            scheduling_policy=config.scheduling_policy,
            max_sessions=config.max_sessions,
            max_num_batched_tokens=runner.max_num_tokens,
            max_live_sessions=store.max_sessions,
            num_live_sessions=store.num_live,
            num_kv_blocks=store.num_blocks,
            free_kv_blocks=store.num_free_blocks,
            block_size=store.block_size,
            max_model_len=runner.max_model_len,
            num_overflowed_sessions=store.num_overflowed,
            clients=clients,
            uptime_s=time.perf_counter() - self._started,
            counters=self.counters,
        )

    def stats(self) -> dict:
        """The counters as per-round and per-step means, for the log."""
        c, store = self.counters, self.runner.sessions
        rounds = max(c.num_rounds, 1)
        return {
            "clients": len(self.clients),
            "rounds": c.num_rounds,
            "sessions_per_round": c.num_round_sessions / rounds,
            "requests_per_round": c.num_round_requests / rounds,
            "draft_tokens_per_round": c.num_round_draft_tokens / rounds,
            "launch_ms_per_round": 1e3 * c.launch_time_s / rounds,
            "round_ms": 1e3 * c.round_time_s / rounds,
            "max_round_ms": 1e3 * c.max_round_time_s,
            "queue_ms_per_step": 1e3 * c.queue_time_s / max(c.num_steps, 1),
            "max_queue_ms": 1e3 * c.max_queue_time_s,
            "max_loop_ms": 1e3 * c.max_loop_time_s,
            "prelaunched_rounds": c.num_prelaunched_rounds,
            "prefix_tokens": c.num_prefix_tokens,
            "live_sessions": store.num_live,
            "free_blocks": store.num_free_blocks,
            "num_blocks": store.num_blocks,
            "overflowed_sessions": store.num_overflowed,
        }

    def _log_stats(self) -> None:
        logger.info("Drafter stats: %s", json.dumps(self.stats()))
