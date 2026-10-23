# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculator whose drafter runs in another process (a drafter service).

The target engine keeps no draft weights and no draft KV: each live request
is a session at the service, opened with its prompt when the request enters
the batch and closed when it leaves. propose() ships the per-row verify
result of the step (see protocol.py) and receives the next drafts; both
transfers are NCCL operations enqueued on the current stream, so the CPU
never waits for the drafter (async scheduling keeps running one step ahead).
Only TP rank 0 talks to the service and broadcasts the drafts.
"""

import os
import socket as pysocket
from typing import Any

import torch
import zmq

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_tp_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.remote.protocol import (
    DRAFTER_RANK,
    TARGET_RANK,
    Hello,
    NewSession,
    Ready,
    StepHeader,
    pack_rows,
    pack_step,
)
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)

READY_TIMEOUT_S = 1800  # the service may still be loading / capturing graphs


class RemoteSpeculator(BaseSpeculator):
    num_query_per_req = 1
    num_extra_query_per_req = 0

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        spec = vllm_config.speculative_config
        assert spec is not None
        if not spec.drafter_endpoint:
            raise ValueError("disaggregated drafting requires drafter_endpoint")
        if spec.method not in ("draft_model", "eagle3", "dflash"):
            raise ValueError(f"disaggregated drafting does not support {spec.method}")
        self.speculative_config = spec
        self.method = spec.method
        self.k = spec.num_speculative_tokens
        self.endpoint: str = spec.drafter_endpoint
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        # Runner-facing attributes that DraftModelSpeculator normally provides.
        self.supports_mm_inputs = False
        self.enable_adaptive_verification = False
        self.draft_token_confidence_probs = None
        self.draft_logits = None  # greedy drafts only
        self.draft_attn_layer_names: set[str] = set()
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_manager = None

        self._tp = get_tp_group()
        self._is_leader = self._tp.rank_in_group == 0
        self._connected = False
        self._zmq_ctx: zmq.Context | None = None
        self._step_socket: zmq.Socket | None = None
        self._comm: PyNcclCommunicator | None = None
        self._target_id = 0
        self._step = 0
        # Session control queued by the runner hooks, flushed with the next
        # header.
        self._pending_new: list[NewSession] = []
        self._pending_released: list[int] = []
        # Slots with a live session on the service. Warm-up batches (no live
        # request, possibly under CUDA graph capture) must not start a round:
        # a captured NCCL send never executes and the service waits forever.
        self._live: set[int] = set()

    # ------------------------------------------------------------ lifecycle

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Called once on every rank after KV-cache setup: a convenient point to
        # connect (the dummy runs before it never reach the service).
        if self._connected:
            return
        if self._is_leader:
            self._connect()
        self._target_id = self._tp.broadcast_object(self._target_id, src=0)
        self._connected = True

    def capture(self) -> None:
        pass

    def load_model(self, target_model) -> None:
        pass

    def prepare_watermarking(self, *args, **kwargs) -> None:
        raise NotImplementedError("disaggregated drafting does not support watermarking")

    def shutdown(self) -> None:
        if self._step_socket is not None:
            self._step_socket.send_pyobj(None)
            self._step_socket.close(linger=1000)
            self._step_socket = None

    # ------------------------------------------------------- runner hooks

    def on_requests_added(self, sessions: list[tuple[int, list[int]]]) -> None:
        """(slot, token ids) of requests that entered the batch this step."""
        self._pending_new.extend(NewSession(s, t) for s, t in sessions)
        self._live.update(s for s, _ in sessions)

    def on_requests_removed(self, slots: list[int]) -> None:
        self._pending_released.extend(slots)
        self._live.difference_update(slots)

    # ---------------------------------------------------------- connection

    def _connect(self) -> None:
        host, port = self.endpoint.rsplit(":", 1)
        ctx = zmq.Context()
        control = ctx.socket(zmq.REQ)
        control.setsockopt(zmq.RCVTIMEO, READY_TIMEOUT_S * 1000)
        control.connect(f"tcp://{host}:{port}")
        mc = self.vllm_config.model_config
        dmc = self.speculative_config.draft_model_config
        hello = Hello(
            method=self.method,
            target_model=mc.model,
            draft_model=dmc.model if dmc is not None else "",
            dtype=str(mc.dtype).removeprefix("torch."),
            max_model_len=mc.max_model_len,
            max_num_seqs=self.max_num_reqs,
            max_num_batched_tokens=self.vllm_config.scheduler_config.max_num_batched_tokens,
            num_speculative_tokens=self.k,
            hidden_size=0 if self.method == "draft_model" else mc.get_hidden_size(),
            num_aux_hidden=3 if self.method == "eagle3" else 0,
            target_label=f"{os.path.basename(mc.model)}@{pysocket.gethostname()}:{os.getpid()}",
        )
        logger.info("Connecting to drafter service at %s: %s", self.endpoint, hello)
        control.send_pyobj(hello)
        try:
            ready: Ready = control.recv_pyobj()
        except zmq.Again as e:
            raise RuntimeError(
                f"drafter service at {self.endpoint} not ready within {READY_TIMEOUT_S} s"
            ) from e
        control.close()
        # A head needs the hidden payload; a draft model takes either: tokens
        # from an engine service, or the step layout (without hidden states)
        # from a dedicated worker.
        allowed = ("tokens", "hidden") if self.method == "draft_model" else ("hidden",)
        if ready.payload not in allowed:
            raise RuntimeError(
                f"drafter service at {self.endpoint} serves payload "
                f"{ready.payload!r}, this target needs one of {allowed}"
            )
        self._hidden = ready.payload == "hidden"
        self._target_id = ready.target_id
        self._step_socket = ctx.socket(zmq.PUSH)
        self._step_socket.connect(f"tcp://{host}:{ready.step_port}")
        group = StatelessProcessGroup.create(
            host=host, port=ready.nccl_port, rank=TARGET_RANK, world_size=2
        )
        self._comm = PyNcclCommunicator(group, device=self.device)
        # NCCL builds its transports on the first transfer in each direction;
        # pay that now rather than on the first step.
        warm = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._comm.send(warm, dst=DRAFTER_RANK)
        self._comm.recv(warm, src=DRAFTER_RANK)
        torch.cuda.synchronize()
        self._zmq_ctx = ctx
        logger.info(
            "Drafter service at %s ready (target_id=%d, payload=%s)",
            self.endpoint,
            ready.target_id,
            ready.payload,
        )

    # ------------------------------------------------------------- propose

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        drafts = torch.zeros((num_reqs, self.k), dtype=torch.int64, device=self.device)
        if dummy_run or is_profile or not self._connected:
            return drafts
        if torch.cuda.is_current_stream_capturing() or not (
            self._live or self._pending_new or self._pending_released
        ):
            return drafts
        if self._hidden:
            # A head needs every target step (its KV is built from the
            # target's hidden states, prompt chunks included).
            has_step = True
        else:
            # Rows that sample this step: every decode row, and a prefill row
            # whose last chunk runs now. The others need no drafts; if none
            # samples, the round trip is skipped (session control still goes
            # out).
            can_sample = ~input_batch.is_prefilling_np | (
                input_batch.num_computed_prefill_tokens_np
                + input_batch.num_scheduled_tokens
                >= input_batch.prefill_len_np
            )
            has_step = bool(can_sample.any())
        if not self._is_leader:
            pass
        elif has_step or self._pending_new or self._pending_released:
            header = StepHeader(
                target_id=self._target_id,
                step=self._step,
                has_step=has_step,
                slots=input_batch.idx_mapping_np[:num_reqs].tolist() if has_step else [],
                k=self.k,
                released=self._pending_released or None,
                new=self._pending_new or None,
            )
            self._pending_new, self._pending_released = [], []
            assert self._step_socket is not None and self._comm is not None
            idx = input_batch.idx_mapping[:num_reqs]
            if has_step and self._hidden:
                n = input_batch.num_tokens
                if self.method == "draft_model":
                    hidden = None
                else:
                    hidden = (
                        torch.cat(aux_hidden_states, dim=-1)
                        if aux_hidden_states
                        else last_hidden_states
                    )[:n].contiguous()
                header.num_tokens = n
                header.num_hidden = 0 if hidden is None else hidden.shape[-1]
                header.query_start_loc = input_batch.query_start_loc_np[: num_reqs + 1].tolist()
                header.num_computed = input_batch.num_computed_tokens_np[:num_reqs].tolist()
                header.is_prefilling = input_batch.is_prefilling_np[:num_reqs].tolist()
                self._step_socket.send_pyobj(header)
                buf = pack_step(
                    input_batch.input_ids[:n],
                    input_batch.positions[:n],
                    input_batch.query_start_loc[: num_reqs + 1],
                    input_batch.seq_lens[:num_reqs],
                    num_sampled[:num_reqs],
                    num_rejected[:num_reqs],
                    last_sampled.view(-1)[idx],
                    next_prefill_tokens[0][idx],
                )
                self._comm.send(buf, dst=DRAFTER_RANK)
                if hidden is not None:
                    self._comm.send(hidden, dst=DRAFTER_RANK)
                self._comm.recv(drafts, src=DRAFTER_RANK)
            elif has_step:
                self._step_socket.send_pyobj(header)
                # KV the target holds after this step: the scheduled tokens
                # minus the rejected drafts (zero for prefill rows).
                num_computed = input_batch.seq_lens[:num_reqs] - num_rejected[:num_reqs]
                bonus = last_sampled.view(-1)[idx]
                buf = pack_rows(num_computed, num_sampled[:num_reqs], bonus)
                self._comm.send(buf, dst=DRAFTER_RANK)
                self._comm.recv(drafts, src=DRAFTER_RANK)
            else:
                self._step_socket.send_pyobj(header)
        self._step += 1
        if self._tp.world_size > 1:
            self._tp.broadcast(drafts, src=0)
        return drafts

    # Kept for parity with DraftModelSpeculator; nothing to observe remotely.
    def observe_verification(self, *args, **kwargs) -> None:
        pass
