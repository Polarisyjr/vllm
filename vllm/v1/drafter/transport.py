# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The tensor channel between a client and an engine: a two-rank NCCL
communicator, the engine at rank 0."""

import os
import socket

import torch

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

# By default NCCL gives every communicator 64 channels for collectives, each
# with its own buffers on the GPU: about 2 GiB on either side of a link over
# InfiniBand. A link only carries sends and receives, and an engine has a link
# per client. NCCL reads the settings once per process, with its first
# communicator, so they also hold for communicators the process opens later.
NCCL_LINK_ENV = {"NCCL_MAX_NCHANNELS": "1", "NCCL_MIN_NCHANNELS": "1"}


def open_link(
    host: str,
    port: int,
    rank: int,
    device: torch.device,
    listen_socket: socket.socket | None = None,
) -> PyNcclCommunicator:
    """Build one side of a link. `host:port` is the rendezvous the engine
    listens on (`listen_socket`, on the engine's side). Waits for the peer."""
    for name, value in NCCL_LINK_ENV.items():
        os.environ.setdefault(name, value)
    group = StatelessProcessGroup.create(
        host=host, port=port, rank=rank, world_size=2, listen_socket=listen_socket
    )
    comm = PyNcclCommunicator(group, device=device)
    if comm.disabled:
        raise RuntimeError("a drafter engine link needs NCCL (PyNccl is disabled)")
    return comm
