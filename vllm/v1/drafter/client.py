# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The target's side of a drafter engine: `DrafterClient`.

    client = DrafterClient("host:7700", device)
    client.describe()                    # what the engine serves and reads
    client.attach(Attach(...))
    pending = client.submit(DraftRequest(sessions, layout), pack_step(...), hidden)
    ...                                  # the stream is free for other work
    reply = client.collect(pending)
    client.release(sessions)
    client.detach()

Neither call waits on the CPU: the transfers are NCCL operations enqueued on
CUDA streams. `submit()` sends on the current stream and receives the reply
on a stream of its own; `collect()` makes the current stream wait for that
reply, so what is enqueued in between runs while the engine drafts.
`DrafterRouter` (router.py) has the same interface over several engines.
"""

import atexit
import contextlib
import os
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import zmq

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger
from vllm.v1.drafter.protocol import (
    CLIENT_RANK,
    ENGINE_RANK,
    Attach,
    Attached,
    Detach,
    DrafterInfo,
    DraftRequest,
    Query,
    Release,
    SetDraftTokens,
    Status,
)
from vllm.v1.drafter.transport import open_link

logger = init_logger(__name__)

# The engine may still be loading or capturing graphs when a target attaches.
ATTACH_TIMEOUT_S = 1800


def _control(
    context: zmq.Context, endpoint: str, message: Any, timeout_s: float
) -> Any:
    """One exchange on the engine's control socket."""
    sock = context.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.connect(f"tcp://{endpoint}")
    try:
        sock.send_pyobj(message)
        return sock.recv_pyobj()
    except zmq.Again as e:
        raise TimeoutError(
            f"drafter engine at {endpoint} did not answer in {timeout_s:.0f} s"
        ) from e
    finally:
        sock.close()


def query_status(endpoint: str, timeout_s: float = 5.0) -> Status:
    """Ask the engine at "host:port" for its Status. The engine answers
    between rounds, within a few milliseconds."""
    return _control(zmq.Context.instance(), endpoint, Query(), timeout_s)


def set_draft_tokens(endpoint: str, num_tokens: int, timeout_s: float = 5.0) -> Status:
    """Have the engine at "host:port" draft at most `num_tokens` tokens per
    session from its next round on."""
    return _control(
        zmq.Context.instance(), endpoint, SetDraftTokens(num_tokens), timeout_s
    )


@dataclass
class PendingDraft:
    """A submitted step whose reply is on its way."""

    # [num_sessions, 2 * k] int32 (protocol.unpack_reply), valid once `landed`.
    reply: torch.Tensor
    landed: torch.cuda.Event


class DrafterClient:
    def __init__(self, endpoint: str, device: torch.device, timeout: float = 60.0):
        self.endpoint = endpoint
        self.device = device
        # Seconds without a sign of life from the engine after which this
        # process gives up and exits (see `_watchdog`). 0 waits forever.
        self.timeout = timeout
        self.client_id = -1
        # What the engine serves (`describe`), and the layers of the target
        # whose hidden states it gets with every step (`use_hidden_layers`).
        self.info: DrafterInfo | None = None
        self.hidden_layers: tuple[int, ...] = ()
        self._linked = False
        self._k = 0
        self._zmq = zmq.Context()
        self._requests: zmq.Socket | None = None
        self._comm: PyNcclCommunicator | None = None
        # Replies are received here, off the stream that runs the target.
        self._reply_stream: torch.cuda.Stream | None = None

    @property
    def attached(self) -> bool:
        return self._linked and self.client_id >= 0

    def describe(self) -> DrafterInfo:
        """What the engine serves. Blocks until the engine is up."""
        if self.info is None:
            logger.info("Waiting for the drafter engine at %s", self.endpoint)
            status: Status = _control(
                self._zmq, self.endpoint, Query(), ATTACH_TIMEOUT_S
            )
            self.info = status.drafter
            if self.info.hidden_layers:
                self.hidden_layers = tuple(self.info.hidden_layers)
        return self.info

    def use_hidden_layers(self, default: tuple[int, ...]) -> tuple[int, ...]:
        """Settle the layers this engine gets hidden states of: the ones it
        names, or `default`, the target's own choice, when it leaves that to
        the target. Returns them; () for an engine that reads none."""
        info = self.describe()
        if info.hidden_layers is not None and not info.hidden_layers:
            self.hidden_layers = tuple(sorted(default))
        return self.hidden_layers

    def attach(self, attach: Attach) -> None:
        """Register with the engine and build the tensor link. Blocks until
        the engine is up; raises if the engine refuses."""
        info = self.describe()
        if info.hidden_layers is not None and not self.hidden_layers:
            raise RuntimeError(
                f"drafter engine at {self.endpoint} reads the target's default "
                "layers: call use_hidden_layers() first"
            )
        attach = replace(attach, hidden_layers=self.hidden_layers)
        host = self.endpoint.rsplit(":", 1)[0]
        logger.info("Attaching to the drafter engine at %s", self.endpoint)
        attached: Attached = _control(
            self._zmq, self.endpoint, attach, ATTACH_TIMEOUT_S
        )
        if attached.client_id < 0:
            raise RuntimeError(
                f"drafter engine at {self.endpoint} refused: {attached.error}"
            )
        self.client_id = attached.client_id
        if self.timeout > 0:
            threading.Thread(
                target=self._watchdog,
                args=(self.client_id,),
                name="drafter-watchdog",
                daemon=True,
            ).start()
        self._requests = self._zmq.socket(zmq.PUSH)
        self._requests.connect(f"tcp://{host}:{attached.request_port}")
        self._comm = open_link(host, attached.nccl_port, CLIENT_RANK, self.device)
        # NCCL builds a transport on the first transfer in each direction;
        # pay for that here rather than on a request.
        warm = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._comm.send(warm, dst=ENGINE_RANK)
        self._comm.recv(warm, src=ENGINE_RANK)
        torch.cuda.synchronize()
        self._reply_stream = torch.cuda.Stream(device=self.device)
        self._linked = True
        self._k = attach.num_speculative_tokens
        atexit.register(self.detach)
        logger.info(
            "Attached to the drafter engine at %s as client %d",
            self.endpoint,
            self.client_id,
        )

    def submit(
        self,
        request: DraftRequest,
        step: torch.Tensor,
        hidden: Mapping[int, torch.Tensor] | None = None,
    ) -> PendingDraft:
        """Send one step to the engine.

        `hidden` holds the target's hidden states of the step's tokens by
        layer, [num_tokens, hidden_size] each; the engine gets the layers it
        reads (`hidden_layers`), none if it reads none.

        The request reaches the engine now, ahead of its payload (`step`,
        protocol.pack_step, and the hidden states), which the current stream
        sends once it has computed it. The engine takes one step of a client
        at a time: the payload of the next step is not sent before the reply
        to this one has landed.
        """
        assert self._comm is not None and self._requests is not None
        assert self._reply_stream is not None
        request.client_id = self.client_id
        self._requests.send_pyobj(request)
        self._comm.send(step, dst=ENGINE_RANK)
        if self.hidden_layers:
            assert hidden is not None, "this engine reads hidden states"
            layers = [hidden[layer] for layer in self.hidden_layers]
            self._comm.send(torch.cat(layers, dim=-1), dst=ENGINE_RANK)
        stream = self._reply_stream
        with torch.cuda.stream(stream):
            reply = torch.empty(
                (len(request.sessions), 2 * self._k),
                dtype=torch.int32,
                device=self.device,
            )
            self._comm.recv(reply, src=ENGINE_RANK, stream=stream)
            landed = torch.cuda.Event()
            landed.record(stream)
        return PendingDraft(reply, landed)

    def collect(self, pending: PendingDraft) -> torch.Tensor:
        """The reply to a submitted step, [num_sessions, 2 * k] int32
        (protocol.unpack_reply). The current stream waits for it."""
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(pending.landed)
        # Allocated on the reply stream, read on this one.
        pending.reply.record_stream(stream)
        return pending.reply

    def release(self, sessions: list[int]) -> None:
        """The engine may drop the KV of these sessions. Sent on the request
        channel, so it is applied after the steps already sent for them."""
        if self._requests is not None:
            self._requests.send_pyobj(Release(sessions, self.client_id))

    def detach(self) -> None:
        if not self.attached:
            return
        client_id, self.client_id = self.client_id, -1
        self._linked = False
        # A reply on its way must land first: the engine cannot take a send
        # back, and a link is only safe to destroy with nothing in flight.
        assert self._reply_stream is not None
        deadline = time.monotonic() + 5
        while not self._reply_stream.query() and time.monotonic() < deadline:
            time.sleep(0.01)
        with contextlib.suppress(TimeoutError):
            _control(self._zmq, self.endpoint, Detach(client_id), timeout_s=5)
        # Free the link's GPU buffers. No thread can be started when the
        # interpreter is already shutting down; the process frees them then.
        if self._comm is not None and self._reply_stream.query():
            with contextlib.suppress(RuntimeError):
                self._comm.destroy()
        self._comm = None

    # ------------------------------------------------------------- watchdog

    def _watchdog(self, client_id: int) -> None:
        """Exit when the engine is gone: for `timeout` it has not finished the
        link or has not answered a status query, or it has dropped this client.

        A dead engine cannot be worked around from here: the receive of its
        answer is already on the GPU stream, with the next forward pass queued
        behind it, and the thread that could raise is blocked on that stream.
        Leaving at least frees the GPU and tells the caller.

        Nothing here may touch CUDA: a CUDA call from this thread while the
        main thread captures a graph invalidates the capture.
        """
        period = min(1.0, self.timeout / 4)
        last_sign = time.monotonic()
        while True:
            time.sleep(period)
            why = ""
            if not self._linked:
                why = "the link is not up"
            else:
                try:
                    status: Status = _control(
                        self._zmq, self.endpoint, Query(client_id), timeout_s=period
                    )
                except TimeoutError:
                    why = "no answer to a status query"
            if self.client_id != client_id:
                return  # detached
            if not why:
                last_sign = time.monotonic()
                if all(c.client_id != client_id for c in status.clients):
                    self._give_up("the engine has dropped this client")
            elif time.monotonic() - last_sign > self.timeout:
                self._give_up(f"{why} after {self.timeout:.0f} s (drafter_timeout)")

    def _give_up(self, why: str) -> None:
        logger.error(
            "Drafter engine at %s: %s. The target cannot continue without it; exiting.",
            self.endpoint,
            why,
        )
        sys.stderr.flush()
        os._exit(1)
