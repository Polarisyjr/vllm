# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-side proposers for disaggregated speculative decoding.

The drafter runs in a separate process, usually on another GPU. These proposers
keep the target runner unchanged: each receives the same arguments as the
in-process proposer's propose(), sends them to the drafter, and returns the
drafted token ids. Three flavours exist, one per upstream proposer:

  RemoteDraftModelProposer  method="draft_model"   (token ids only)
  RemoteEagleProposer       method="eagle"/"eagle3" (plus target hidden states)
  RemoteDFlashProposer      method="dflash"         (plus target hidden states)

The drafter mirrors the target's KV cache block table: it allocates the same
block size and at least as many blocks, and writes draft KV into the block ids
the target scheduler assigned. It therefore needs no request lifecycle state,
and every propose() call carries everything one draft round needs.

Wire protocol, per propose() call:
  1. StepHeader over ZMQ PUSH/PULL (sizes and CPU-side scalars).
  2. One int32 tensor over NCCL, target -> drafter (see pack_step).
  3. For hidden-state drafters, the target hidden states [num_tokens, H] over
     NCCL, target -> drafter, in the target's dtype.
  4. Drafted token ids, int64 [num_reqs, k], over NCCL, drafter -> target.
NCCL ops are enqueued on the current CUDA stream, so propose() does not block
the target's CPU; the next step's use of the drafts is ordered after the receive.

With tensor parallelism only TP rank 0 talks to the drafter (every rank holds
the same token ids and hidden states); it broadcasts the drafts to the other
ranks over the TP group.

Same-GPU drafter (the drafter process shares the leader's GPU): NCCL cannot
pair two ranks on one device, and copies are pointless there, so the two
processes share device buffers through CUDA IPC instead and order their
streams with interprocess CUDA events:
  1. StepHeader over ZMQ (ipc:// socket).
  2. The packed int32 tensor is written into a shared buffer on the target's
     stream, then `ev_in` is recorded (owned by the target, imported by the
     drafter), and the header is sent.
  3. The drafter waits for `ev_in` on its stream, drafts, writes the drafts
     into the shared output buffer, records `ev_out` (owned by the drafter)
     and sends a one-byte ack.
  4. The target waits for the ack (the drafter has *enqueued* its work, not
     finished it), makes its stream wait for `ev_out` and copies the drafts
     out. The only CPU stall on the target is the drafter's launch time.
The drafter chooses this transport when the target's device UUID matches its
own. Only token-id drafters (draft_model) use it so far.
"""

from __future__ import annotations

import atexit
import os
import socket
import tempfile
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import zmq
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_tp_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer

logger = init_logger(__name__)

DRAFTER_RANK = 0
TARGET_RANK = 1
DRAFTER_READY_TIMEOUT_S = 900


@dataclass
class DrafterHello:
    """Sent once by the target when its attention backend is first set up."""

    method: str
    target_model: str
    draft_model: str
    dtype: str
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    block_size: int
    num_speculative_tokens: int
    # The target's RoPE layout; DFlash heads must match it.
    target_rope_is_neox_style: bool | None = None
    # UUID of the leader's GPU; a drafter on the same device switches to the
    # CUDA IPC transport.
    device_uuid: str | None = None
    # Free-form label of this target (drafter-service logs and stats).
    target_label: str = ""


@dataclass
class DrafterReady:
    step_port: int
    nccl_port: int
    # The drafter sizes its KV cache from its own GPU memory. Target block ids
    # must stay below this.
    num_gpu_blocks: int
    # Set for the same-GPU transport: exported shared buffers, the drafter's
    # event handle and the ZMQ endpoints (see IpcChannel).
    ipc: dict | None = None
    # Drafter service: the drafter keeps its own KV cache per request (keyed
    # by target_id and request slot), so no block table crosses the wire and
    # the target's KV cache is not capped. Several targets may share it.
    owns_kv: bool = False
    target_id: int = 0


@dataclass
class IpcSetup:
    """Second control message in the same-GPU handshake: the target's event."""

    ev_in: bytes


def device_uuid(device: torch.device | int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def export_tensor(t: torch.Tensor) -> tuple:
    """CUDA IPC handle of a device tensor (torch.multiprocessing's encoding);
    the exporter must keep `t` alive."""
    fn, args = reduce_tensor(t)
    assert fn is rebuild_cuda_tensor, fn
    return tuple(args)


def import_tensor(args: tuple, device_index: int) -> torch.Tensor:
    """Open an exported tensor on `device_index` (the importer's numbering;
    both processes may see the GPU under different CUDA_VISIBLE_DEVICES)."""
    args = list(args)
    args[6] = device_index  # storage_device
    return rebuild_cuda_tensor(*args)


def ipc_endpoints(port: int) -> dict[str, str]:
    base = os.path.join(tempfile.gettempdir(), f"specpilot_drafter_{port}")
    return {"step": f"ipc://{base}.step", "ack": f"ipc://{base}.ack"}


def max_packed_numel(max_num_tokens: int, max_num_seqs: int, max_model_len: int,
                     block_size: int, num_speculative_tokens: int) -> int:
    b, n = max_num_seqs, max_num_tokens
    cols = cdiv(max_model_len + num_speculative_tokens + 1, block_size)
    return 2 * n + b + (b + 1) + b + b + b + b * cols


@dataclass
class IpcChannel:
    """Target-side view of the same-GPU transport."""

    inbuf: torch.Tensor
    outbuf: torch.Tensor
    ev_in: torch.cuda.Event
    ev_out: torch.cuda.Event
    step_socket: zmq.Socket
    ack_socket: zmq.Socket
    steps: int = field(default=0)


@dataclass
class DrafterLink:
    """One drafter connection. A target may hold several (comma-separated
    ``drafter_endpoint``) and binds each request to one of them on first
    sight (``SamplingParams.extra_args["drafter"]``: endpoint index, or -1 for
    no drafter; default: the SPECPILOT_DRAFTER_BINDING policy, "0" or "rr")."""

    endpoint: str
    step_socket: zmq.Socket | None = None
    comm: PyNcclCommunicator | None = None
    ipc: IpcChannel | None = None
    owns_kv: bool = False
    target_id: int = 0
    num_blocks: int = 0
    released: list[int] = field(default_factory=list)
    num_reqs_bound: int = 0


@dataclass
class StepHeader:
    num_reqs: int
    num_tokens: int
    num_block_cols: int
    num_speculative_tokens: int
    max_query_len: int
    max_seq_len: int
    has_rejected: bool
    query_start_loc_cpu: list[int]
    # Width of the hidden-state tensor that follows the int32 tensor; 0 when
    # the drafter takes token ids only.
    num_hidden: int = 0
    # Per request, the number of draft tokens the target verified this step, or
    # -1 for requests that verified none (prompt chunks, plain decodes). Hybrid
    # drafters classify their recurrent-state rows with it.
    num_draft_tokens: list[int] | None = None
    # Per request, the number of tokens computed before this step (the row's
    # first position). Hybrid drafters drop the first token of rows that
    # continue a sequence, which they consumed as the previous step's bonus.
    num_computed_tokens: list[int] | None = None
    # Drafter service: which target sent the step, the request slots released
    # since the previous step (the drafter frees their KV) and the slots of
    # the rows in batch order.
    target_id: int = 0
    released_slots: list[int] | None = None
    req_slots: list[int] | None = None


def packed_numel(h: StepHeader) -> int:
    b, n = h.num_reqs, h.num_tokens
    return (
        2 * n  # token ids, positions
        + b  # next token ids
        + (b + 1)  # query_start_loc
        + b  # seq_lens
        + (b if h.has_rejected else 0)  # num_rejected_tokens
        + b  # request slots
        + b * h.num_block_cols  # block table
    )


def pack_step(
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    next_token_ids: torch.Tensor,
    cad: CommonAttentionMetadata,
    num_rejected_tokens: torch.Tensor | None,
    req_slots: torch.Tensor,
    num_block_cols: int,
) -> torch.Tensor:
    parts = [
        target_token_ids.to(torch.int32),
        target_positions.to(torch.int32),
        next_token_ids.to(torch.int32),
        cad.query_start_loc.to(torch.int32),
        cad.seq_lens.to(torch.int32),
    ]
    if num_rejected_tokens is not None:
        parts.append(num_rejected_tokens.to(torch.int32))
    parts.append(req_slots.to(torch.int32))
    parts.append(cad.block_table_tensor[:, :num_block_cols].reshape(-1))
    return torch.cat(parts)


def unpack_step(buf: torch.Tensor, h: StepHeader) -> dict[str, torch.Tensor | None]:
    b, n = h.num_reqs, h.num_tokens
    sizes = [n, n, b, b + 1, b, b if h.has_rejected else 0, b, b * h.num_block_cols]
    (
        token_ids,
        positions,
        next_token_ids,
        query_start_loc,
        seq_lens,
        rejected,
        req_slots,
        block_table,
    ) = torch.split(buf, sizes)
    return dict(
        target_token_ids=token_ids,
        target_positions=positions.to(torch.int64),
        next_token_ids=next_token_ids,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        num_rejected_tokens=rejected if h.has_rejected else None,
        req_slots=req_slots,
        block_table=block_table.view(b, h.num_block_cols),
    )


class RemoteDrafterMixin:
    """Connection and wire protocol shared by the remote proposers.

    Mixed in ahead of an upstream proposer class, so these methods override the
    upstream model loading, attention-backend setup and propose().
    """

    def _init_remote(self, runner=None) -> None:
        spec = self.speculative_config
        if not spec.drafter_endpoint:
            raise ValueError("disaggregated drafting requires drafter_endpoint")
        if spec.disable_padded_drafter_batch:
            raise ValueError("disaggregated drafting requires the padded drafter batch")
        if self._enable_probabilistic_draft_probs:
            raise ValueError("disaggregated drafting supports greedy drafts only")
        if self.uses_mrope and self.method != "draft_model":
            raise ValueError("disaggregated drafting does not support M-RoPE heads")
        self.endpoint: str = spec.drafter_endpoint
        self._runner = runner
        # Request slots: a small stable integer per live request, assigned on
        # first sight and released when the request leaves the batch. Hybrid
        # drafters index their recurrent-state rows with it.
        self._req_slots: dict[str, int] = {}
        self._free_slots: list[int] = list(range(self.max_batch_size))[::-1]
        self._slot_buf = torch.zeros(self.max_batch_size, dtype=torch.int32)
        self._num_draft_tokens: list[int] | None = None
        self._tp = get_tp_group()
        self._is_leader = self._tp.rank_in_group == 0
        self._connected = False
        self._drafter_num_blocks = 0
        self._owns_kv = False
        self._target_id = 0
        self._released_slots: list[int] = []
        self._links: list[DrafterLink] = []
        self._req_link: dict[str, int] = {}
        self._slot_link: dict[int, int] = {}
        self._row_links: list[int] = []
        self._rr = 0
        self._binding = os.environ.get("SPECPILOT_DRAFTER_BINDING", "0")
        self._target_rope_is_neox_style: bool | None = None
        self._zmq_ctx: zmq.Context | None = None
        self._step_socket: zmq.Socket | None = None
        self._comm: PyNcclCommunicator | None = None
        self._ipc: IpcChannel | None = None

    # ------------------------------------------------------ runner hooks

    def load_model(self, target_model: nn.Module) -> None:
        # Weights live in the drafter process. Leave `self.model` unset so the
        # runner does not wrap or inspect it, and register no draft KV layers so
        # the target's KV cache holds target layers only.
        from vllm.model_executor.models.qwen3_dflash import (
            dflash_target_rope_is_neox_style,
        )

        self._draft_attn_layer_names = set()
        self._target_rope_is_neox_style = dflash_target_rope_is_neox_style(
            target_model
        )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        pass

    def _ensure_connected(self, block_size: int) -> None:
        if self._connected:
            return
        self.block_size = block_size
        if self._is_leader:
            self._connect()
        # Every rank must take part in the broadcast.
        self._drafter_num_blocks, self._owns_kv, self._target_id = (
            self._tp.broadcast_object(
                (self._drafter_num_blocks, self._owns_kv, self._target_id), src=0
            )
        )
        self._connected = True

    def cap_available_kv_memory(self, available: int, kv_cache_specs: dict) -> int:
        """Cap the target's KV cache memory so it allocates no more blocks
        than the drafter holds (called from the worker's memory profiling)."""
        self._ensure_connected(self._attention_block_size(kv_cache_specs))
        if self._owns_kv:
            return available
        # Size the cache the way the worker will, so hybrid targets (whose
        # layers share KV tensors across groups) get the right block count.
        num_blocks = get_kv_cache_configs(
            self.vllm_config, [dict(kv_cache_specs)], [available]
        )[0].num_blocks
        if num_blocks > self._drafter_num_blocks:
            cap = available * self._drafter_num_blocks // num_blocks
            logger.info(
                "Capping target KV cache memory from %.2f to %.2f GiB (%d -> %d "
                "blocks): the drafter at %s holds %d blocks",
                available / 2**30,
                cap / 2**30,
                num_blocks,
                self._drafter_num_blocks,
                self.endpoint,
                self._drafter_num_blocks,
            )
            return cap
        return available

    @staticmethod
    def _attention_block_size(kv_cache_specs: dict) -> int:
        sizes = {
            spec.block_size
            for spec in kv_cache_specs.values()
            if not isinstance(spec, MambaSpec)
        }
        assert len(sizes) == 1, f"attention block sizes {sizes}"
        return sizes.pop()

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        # Called once for the CUDA-graph memory profiling KV cache and again for
        # the real one. Connect on the first call; check capacity on each.
        # The drafter mirrors the attention group's block table (hybrid targets
        # also have Mamba groups); the runner hands us that group's metadata.
        groups = kv_cache_config.kv_cache_groups
        attn_gids = [
            i for i, g in enumerate(groups) if not isinstance(g.kv_cache_spec, MambaSpec)
        ]
        assert len(attn_gids) == 1, f"attention KV cache groups {attn_gids}"
        self.kv_cache_gid = attn_gids[0]
        if kernel_block_sizes:
            block_size = kernel_block_sizes[self.kv_cache_gid]
        else:
            block_size = groups[self.kv_cache_gid].kv_cache_spec.block_size
        self._ensure_connected(block_size)
        assert block_size == self.block_size, (block_size, self.block_size)
        if self._owns_kv:
            return
        if kv_cache_config.num_blocks > self._drafter_num_blocks:
            raise ValueError(
                f"Target KV cache has {kv_cache_config.num_blocks} blocks but the "
                f"drafter at {self.endpoint} holds only {self._drafter_num_blocks}. "
                "Give the drafter more memory or lower the target's "
                "gpu_memory_utilization."
            )

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def dummy_run(self, *args, **kwargs) -> None:
        pass

    def prepare_inputs_padded(self, common_attn_metadata, spec_decode_metadata,
                              valid_sampled_tokens_count):
        # The runner calls this right before propose() when drafts were
        # verified; remember which requests carried how many.
        self._num_draft_tokens = list(spec_decode_metadata.num_draft_tokens)
        return super().prepare_inputs_padded(
            common_attn_metadata, spec_decode_metadata, valid_sampled_tokens_count
        )

    def _bind(self, rid: str) -> int:
        """Drafter link index for a request seen for the first time."""
        n = len(self._links)
        choice = None
        req = self._runner.requests.get(rid)
        sp = getattr(req, "sampling_params", None)
        extra = getattr(sp, "extra_args", None) or {}
        if "drafter" in extra:
            choice = extra["drafter"]
            if choice in (None, "none"):
                choice = -1
        elif self._binding == "rr":
            choice = self._rr % n
            self._rr += 1
        else:
            choice = int(self._binding)
        choice = int(choice)
        if choice >= n:
            raise ValueError(f"request {rid} asks for drafter {choice}, only {n} configured")
        if choice >= 0:
            self._links[choice].num_reqs_bound += 1
        return choice

    def _request_slots(self, num_reqs: int) -> torch.Tensor:
        """Stable per-request slot ids for the current batch order; also
        records each row's drafter link (binding on first sight) and queues
        released slots for the link that served them."""
        req_ids = self._runner.input_batch.req_ids[:num_reqs]
        live = set(req_ids)
        for rid in [r for r in self._req_slots if r not in live]:
            slot = self._req_slots.pop(rid)
            self._free_slots.append(slot)
            self._released_slots.append(slot)
            li = self._slot_link.pop(slot, 0)
            self._req_link.pop(rid, None)
            if self._links and li >= 0:
                self._links[li].released.append(slot)
        slots = self._slot_buf
        row_links = []
        for i, rid in enumerate(req_ids):
            slot = self._req_slots.get(rid)
            if slot is None:
                slot = self._req_slots[rid] = self._free_slots.pop()
                li = self._bind(rid) if self._links else 0
                self._req_link[rid] = li
                self._slot_link[slot] = li
            slots[i] = slot
            row_links.append(self._req_link.get(rid, 0))
        self._row_links = row_links
        self._slots_cpu = slots[:num_reqs].tolist()
        return slots[:num_reqs].to(self.device, non_blocking=True)

    # ------------------------------------------------------ connection

    def _connect(self) -> None:
        endpoints = [e.strip() for e in self.endpoint.split(",") if e.strip()]
        for ep in endpoints:
            self._links.append(self._connect_one(ep))
        first = self._links[0]
        # Compatibility with the single-link code paths (same-GPU IPC).
        self._comm, self._step_socket, self._ipc = first.comm, first.step_socket, first.ipc
        self._owns_kv = all(l.owns_kv for l in self._links)
        if any(l.owns_kv for l in self._links) and not self._owns_kv:
            raise ValueError("mixing drafter services and block-mirroring drafters")
        self._target_id = first.target_id
        self._drafter_num_blocks = min(l.num_blocks for l in self._links)
        atexit.register(self.shutdown)

    def _connect_one(self, endpoint: str) -> DrafterLink:
        host, port = endpoint.rsplit(":", 1)
        ctx = zmq.Context()
        control = ctx.socket(zmq.REQ)
        # The drafter loads and captures CUDA graphs before replying.
        control.setsockopt(zmq.RCVTIMEO, DRAFTER_READY_TIMEOUT_S * 1000)
        control.connect(f"tcp://{host}:{port}")
        hello = DrafterHello(
            method=self.method,
            target_model=self.vllm_config.model_config.model,
            draft_model=self.draft_model_config.model,
            dtype=str(self.draft_model_config.dtype).removeprefix("torch."),
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_batch_size,
            max_num_batched_tokens=self.max_num_tokens,
            block_size=self.block_size,
            num_speculative_tokens=self.num_speculative_tokens,
            target_rope_is_neox_style=self._target_rope_is_neox_style,
            device_uuid=device_uuid(self.device),
            target_label=f"{os.path.basename(self.vllm_config.model_config.model)}"
            f"@{socket.gethostname()}:{os.getpid()}",
        )
        logger.info("Connecting to drafter at %s: %s", endpoint, hello)
        control.send_pyobj(hello)
        try:
            ready: DrafterReady = control.recv_pyobj()
        except zmq.Again as e:
            raise RuntimeError(
                f"Drafter at {endpoint} did not become ready within "
                f"{DRAFTER_READY_TIMEOUT_S} s"
            ) from e
        link = DrafterLink(endpoint=endpoint, owns_kv=ready.owns_kv,
                           target_id=ready.target_id, num_blocks=ready.num_gpu_blocks)
        if ready.ipc is not None:
            if self._links:
                raise ValueError("the same-GPU IPC drafter must be the only endpoint")
            self._connect_ipc(ctx, control, ready.ipc)
            control.close()
            self._zmq_ctx = ctx
            link.ipc = self._ipc
            logger.info(
                "Drafter at %s shares this GPU: CUDA IPC transport, %d KV blocks",
                endpoint,
                ready.num_gpu_blocks,
            )
            return link
        control.close()

        step_socket = ctx.socket(zmq.PUSH)
        step_socket.connect(f"tcp://{host}:{ready.step_port}")
        group = StatelessProcessGroup.create(
            host=host, port=ready.nccl_port, rank=TARGET_RANK, world_size=2
        )
        comm = PyNcclCommunicator(group, device=self.device)
        # NCCL sets up its transports on the first send/recv in each direction,
        # which costs hundreds of ms; pay that now rather than on the first step.
        warm = torch.zeros(1, dtype=torch.int32, device=self.device)
        comm.send(warm, dst=DRAFTER_RANK)
        comm.recv(warm, src=DRAFTER_RANK)
        torch.cuda.synchronize()
        self._zmq_ctx = ctx
        link.step_socket, link.comm = step_socket, comm
        logger.info(
            "Drafter at %s is ready with %d KV blocks (owns_kv=%s, target_id=%d)",
            endpoint,
            ready.num_gpu_blocks,
            ready.owns_kv,
            ready.target_id,
        )
        return link

    def _connect_ipc(self, ctx: zmq.Context, control: zmq.Socket, ipc: dict) -> None:
        if self.pass_hidden_states_to_model:
            raise ValueError(
                "the same-GPU (CUDA IPC) transport supports token-id drafters only"
            )
        index = self.device.index
        if index is None:
            index = torch.cuda.current_device()
        with torch.cuda.device(index):
            inbuf = import_tensor(ipc["inbuf"], index)
            outbuf = import_tensor(ipc["outbuf"], index)
            ev_out = torch.cuda.Event.from_ipc_handle(index, ipc["ev_out"])
            ev_in = torch.cuda.Event(interprocess=True)
            control.send_pyobj(IpcSetup(ev_in=ev_in.ipc_handle()))
            assert control.recv_pyobj() is True
        step_socket = ctx.socket(zmq.PUSH)
        step_socket.connect(ipc["endpoints"]["step"])
        ack_socket = ctx.socket(zmq.PULL)
        ack_socket.connect(ipc["endpoints"]["ack"])
        self._ipc = IpcChannel(inbuf, outbuf, ev_in, ev_out, step_socket, ack_socket)
        self._step_socket = step_socket

    def _propose_ipc(self, header: StepHeader, buf: torch.Tensor,
                     drafts: torch.Tensor) -> None:
        ipc = self._ipc
        assert ipc is not None
        n = buf.numel()
        if n > ipc.inbuf.numel():
            raise RuntimeError(
                f"step needs {n} packed ints, shared buffer holds {ipc.inbuf.numel()}"
            )
        stream = torch.cuda.current_stream(self.device)
        ipc.inbuf[:n].copy_(buf)
        ipc.ev_in.record(stream)
        ipc.step_socket.send_pyobj(header)
        if header.num_speculative_tokens > 0:
            ipc.ack_socket.recv()
            stream.wait_event(ipc.ev_out)
            drafts.copy_(ipc.outbuf[: header.num_reqs, : header.num_speculative_tokens])
        ipc.steps += 1

    # ------------------------------------------------------ propose

    def _header(self, link: DrafterLink, num_reqs: int, num_tokens: int, k: int,
                num_block_cols: int, max_query_len: int, max_seq_len: int,
                has_rejected: bool, qsl_cpu: list[int], num_hidden: int,
                num_draft_tokens: list[int] | None, num_computed: list[int],
                req_slots: list[int]) -> StepHeader:
        released, link.released = link.released, []
        return StepHeader(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            num_block_cols=num_block_cols,
            num_speculative_tokens=k,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            has_rejected=has_rejected,
            query_start_loc_cpu=qsl_cpu,
            num_hidden=num_hidden,
            num_draft_tokens=num_draft_tokens,
            num_computed_tokens=num_computed,
            target_id=link.target_id,
            released_slots=released or None,
            req_slots=req_slots if link.owns_kv else None,
        )

    def _send(self, link: DrafterLink, header: StepHeader, buf: torch.Tensor,
              hidden: torch.Tensor | None) -> None:
        assert link.comm is not None and link.step_socket is not None
        link.step_socket.send_pyobj(header)
        link.comm.send(buf, dst=DRAFTER_RANK)
        if hidden is not None:
            link.comm.send(hidden.contiguous(), dst=DRAFTER_RANK)

    def propose(
        self,
        num_speculative_tokens: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        if target_positions.dim() == 2:
            # M-RoPE target (e.g. Qwen3.5): text-only input carries the same
            # position on all three rows.
            target_positions = target_positions[0]
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        cad = common_attn_metadata
        num_reqs = cad.batch_size()
        k = num_speculative_tokens
        num_draft_tokens = self._num_draft_tokens
        self._num_draft_tokens = None
        if num_rejected_tokens_gpu is None:
            num_draft_tokens = None
        elif num_draft_tokens is None:
            raise RuntimeError("padded drafter inputs without prepare_inputs_padded")
        else:
            assert len(num_draft_tokens) == num_reqs, (len(num_draft_tokens), num_reqs)
            num_draft_tokens = [n if n > 0 else -1 for n in num_draft_tokens]
        # Columns covering every position the drafter writes this round: the
        # target tokens, the bonus token and the draft steps.
        num_block_cols = min(
            cad.block_table_tensor.shape[1],
            cdiv(cad.max_seq_len + k + 1, self.block_size),
        )
        if self._owns_kv:
            num_block_cols = 0
        send_hidden = self.pass_hidden_states_to_model
        hidden = target_hidden_states if send_hidden else None
        num_hidden = target_hidden_states.shape[-1] if send_hidden else 0
        req_slots = self._request_slots(num_reqs)
        qsl_cpu = cad.query_start_loc_cpu[: num_reqs + 1].tolist()
        num_computed = self._runner.input_batch.num_computed_tokens_cpu[:num_reqs].tolist()
        drafts = torch.empty((num_reqs, k), dtype=torch.int64, device=self.device)
        if not self._is_leader:
            pass
        elif self._ipc is not None:
            link = self._links[0]
            header = self._header(link, num_reqs, target_token_ids.shape[0], k,
                                  num_block_cols, cad.max_query_len, cad.max_seq_len,
                                  num_rejected_tokens_gpu is not None, qsl_cpu, 0,
                                  num_draft_tokens, num_computed, self._slots_cpu)
            buf = pack_step(target_token_ids, target_positions, next_token_ids, cad,
                            num_rejected_tokens_gpu, req_slots, num_block_cols)
            self._propose_ipc(header, buf, drafts)
        else:
            groups: dict[int, list[int]] = {}
            for r, li in enumerate(self._row_links):
                groups.setdefault(li, []).append(r)
            if len(groups) == 1 and -1 not in groups:
                link = self._links[next(iter(groups))]
                header = self._header(link, num_reqs, target_token_ids.shape[0], k,
                                      num_block_cols, cad.max_query_len, cad.max_seq_len,
                                      num_rejected_tokens_gpu is not None, qsl_cpu,
                                      num_hidden, num_draft_tokens, num_computed,
                                      self._slots_cpu)
                buf = pack_step(target_token_ids, target_positions, next_token_ids, cad,
                                num_rejected_tokens_gpu, req_slots, num_block_cols)
                self._send(link, header, buf, hidden)
                if k > 0:
                    link.comm.recv(drafts, src=DRAFTER_RANK)
            else:
                # Rows bound to different drafters (or to none): one sub-step
                # per drafter, sent before any draft is awaited so the
                # drafters work in parallel. Unbound rows get filler drafts
                # (token 0) that the verifier rejects: async scheduling has
                # already reserved k slots for them.
                drafts.zero_()
                pending = []
                for li, rows in groups.items():
                    if li < 0:
                        continue
                    link = self._links[li]
                    rows_t = torch.tensor(rows, dtype=torch.int64, device=self.device)
                    tok_idx = []
                    sub_qsl = [0]
                    sub_computed = []
                    max_q = 0
                    max_s = 0
                    for r in rows:
                        a, b = qsl_cpu[r], qsl_cpu[r + 1]
                        tok_idx.extend(range(a, b))
                        sub_qsl.append(sub_qsl[-1] + b - a)
                        sub_computed.append(num_computed[r])
                        max_q = max(max_q, b - a)
                        max_s = max(max_s, num_computed[r] + b - a)
                    tok_t = torch.tensor(tok_idx, dtype=torch.int64, device=self.device)
                    sub_qsl_t = torch.tensor(sub_qsl, dtype=torch.int32, device=self.device)
                    parts = [
                        target_token_ids[tok_t].to(torch.int32),
                        target_positions[tok_t].to(torch.int32),
                        next_token_ids[rows_t].to(torch.int32),
                        sub_qsl_t,
                        cad.seq_lens[rows_t].to(torch.int32),
                    ]
                    if num_rejected_tokens_gpu is not None:
                        parts.append(num_rejected_tokens_gpu[rows_t].to(torch.int32))
                    parts.append(req_slots[rows_t].to(torch.int32))
                    parts.append(cad.block_table_tensor[rows_t, :num_block_cols].reshape(-1))
                    buf = torch.cat(parts)
                    header = self._header(
                        link, len(rows), len(tok_idx), k, num_block_cols, max_q, max_s,
                        num_rejected_tokens_gpu is not None, sub_qsl, num_hidden,
                        [num_draft_tokens[r] for r in rows] if num_draft_tokens else None,
                        sub_computed, [self._slots_cpu[r] for r in rows])
                    self._send(link, header, buf,
                               hidden[tok_t] if hidden is not None else None)
                    pending.append((link, rows_t))
                for link, rows_t in pending:
                    if k > 0:
                        sub = torch.empty((rows_t.numel(), k), dtype=torch.int64,
                                          device=self.device)
                        link.comm.recv(sub, src=DRAFTER_RANK)
                        drafts[rows_t] = sub
        if k > 0 and self._tp.world_size > 1:
            self._tp.broadcast(drafts, src=0)
        return drafts

    def shutdown(self) -> None:
        for link in self._links:
            if link.step_socket is not None:
                link.step_socket.send_pyobj(None)
                link.step_socket.close(linger=1000)
                link.step_socket = None
        self._step_socket = None


class RemoteDraftModelProposer(RemoteDrafterMixin, DraftModelProposer):
    """DraftModelProposer whose draft model lives in a drafter process."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        if self.use_heterogeneous_vocab:
            raise ValueError(
                "disaggregated drafting does not support use_heterogeneous_vocab"
            )
        self._init_remote(runner)

    def _raise_if_draft_tp_mismatch(self):
        # The draft model runs in the drafter process with its own parallelism.
        pass

    def _raise_if_mrope(self):
        # The drafter feeds its own positions (M-RoPE drafts such as Qwen3.5
        # get the 1-D positions repeated on three rows).
        pass


class RemoteEagleProposer(RemoteDrafterMixin, EagleProposer):
    """EagleProposer (EAGLE / EAGLE3) whose draft head lives in a drafter process.

    The target still produces the auxiliary hidden states the head consumes;
    they cross the wire with each step.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        self._init_remote(runner)


class RemoteDFlashProposer(RemoteDrafterMixin, DFlashProposer):
    """DFlashProposer whose draft head lives in a drafter process."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        self._init_remote(runner)
