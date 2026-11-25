# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port
from vllm.v1.drafter.protocol import (
    CLIENT_RANK,
    ENGINE_RANK,
    Attach,
    DraftRequest,
    Release,
)
from vllm.v1.drafter.transport import open_link

logger = init_logger(__name__)


@dataclass
class Client:
    """An attached target engine, as the drafter engine sees it."""

    id: int
    attach: Attach
    # Messages the engine has not taken yet, in the order they were sent.
    inbox: deque[DraftRequest | Release] = field(default_factory=deque)
    # session id (the client's) -> row
    sessions: dict[int, int] = field(default_factory=dict)
    # The tensor channel, and the CUDA stream its payloads are received on.
    # Set by `link`, which runs on a thread: `linked` or `failed` flips last.
    comm: PyNcclCommunicator | None = None
    stream: torch.cuda.Stream | None = None
    linked: bool = False
    failed: bool = False
    # Steps drafted, and the time their payloads waited for a round.
    num_steps: int = 0
    queue_s: float = 0.0
    # When the client last sent a request or a heartbeat (a Query with its
    # id), and whether it sends heartbeats at all.
    last_seen: float = field(default_factory=time.perf_counter)
    heartbeats: bool = False


def listen(host: str) -> tuple[socket.socket, int]:
    """Socket and port of the rendezvous of a client's NCCL link. It must
    listen before the client is told where it is."""
    port = get_open_port()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen()
    return listener, port


def link(
    client: Client,
    listener: socket.socket,
    host: str,
    port: int,
    device: torch.device,
) -> None:
    """Build the NCCL link to a client and mark the client linked.

    Takes most of a second and waits for the client, so the engine runs it on
    a thread and goes on serving the others meanwhile.
    """
    t0 = time.perf_counter()
    try:
        comm = open_link(host, port, ENGINE_RANK, device, listen_socket=listener)
        # NCCL builds a transport on the first transfer in each direction.
        # On the client's own stream, like its payloads.
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            warm = torch.zeros(1, dtype=torch.int32, device=device)
            comm.recv(warm, src=CLIENT_RANK, stream=stream)
            comm.send(warm, dst=CLIENT_RANK, stream=stream)
        stream.synchronize()
    except Exception:
        logger.exception(
            "Linking client %d (%s) failed", client.id, client.attach.label
        )
        client.failed = True
        return
    client.comm, client.stream = comm, stream
    client.linked = True
    logger.info(
        "Client %d linked in %.0f ms", client.id, 1e3 * (time.perf_counter() - t0)
    )
