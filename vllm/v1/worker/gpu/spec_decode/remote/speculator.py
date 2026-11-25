# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculator whose drafters are drafter engines in other processes.

The target keeps no draft weights and no draft KV, and does not know what
kind of drafter it has: every step, propose() hands the step (and the hidden
states the engines asked for) to its drafter engines through a
`DrafterRouter` (see vllm/v1/drafter) and gets draft tokens and their
acceptance confidences back. The engines may be of different kinds; a
request is drafted by the one it was paired with (`DRAFTER_ARG`). The
transfers are NCCL operations enqueued on CUDA streams, so the CPU never
waits for the drafters and async scheduling keeps running a step ahead.
"""

import os
import socket
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsEagle3, supports_eagle3
from vllm.sampling_params import SamplingParams
from vllm.v1.drafter.protocol import (
    Attach,
    DrafterInfo,
    DraftRequest,
    StepLayout,
    pack_step,
    unpack_reply,
)
from vllm.v1.drafter.router import DrafterRouter
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)

# SamplingParams.extra_args key (`vllm_xargs` of an API request): the
# "host:port" of the engine that drafts for the request, one of the target's
# `drafter_endpoint`s. Without it the target picks one by load.
DRAFTER_ARG = "drafter"
# The number of tokens to draft for the request, at most; without it as many
# as its engine drafts.
DRAFT_TOKENS_ARG = "draft_tokens"


class RemoteSpeculator(BaseSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        spec = vllm_config.speculative_config
        assert spec is not None and spec.drafter_endpoint
        if spec.draft_sample_method != "greedy":
            raise ValueError("drafter_endpoint supports greedy drafts only")
        if vllm_config.parallel_config.tensor_parallel_size > 1:
            # One rank would talk to the engines and broadcast the drafts.
            raise NotImplementedError("drafter_endpoint with tensor parallelism")
        self.speculative_config = spec
        self.num_speculative_steps = spec.num_speculative_tokens
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        # What the model runner reads on a local speculator.
        self.supports_mm_inputs = False
        self.enable_adaptive_verification = spec.enable_adaptive_verification
        self.draft_token_confidence_probs = torch.ones(
            max_num_reqs, self.num_speculative_steps, dtype=torch.float32, device=device
        )
        self.draft_logits = None
        self.pcp_manager = None

        endpoints = spec.drafter_endpoint
        if isinstance(endpoints, str):
            endpoints = [endpoints]
        self.drafters = DrafterRouter(
            list(endpoints), device, timeout=spec.drafter_timeout
        )
        for endpoint, info in zip(endpoints, self.drafters.describe()):
            self._check(endpoint, info)
        # Whether the target has to export hidden states for its engines,
        # and of which layers (`set_aux_hidden_state_layers`).
        self.needs_hidden_states = any(
            info.hidden_layers is not None for info in self.drafters.describe()
        )
        self._hidden_layers: tuple[int, ...] = ()
        # Request slot -> tokens the target holds KV for at the request's
        # first step, sent to the engine with that step.
        self._prefixes: dict[int, list[int]] = {}
        # Request slot -> its DRAFT_TOKENS_ARG.
        self._draft_tokens: dict[int, int] = {}
        # See `on_step_sized`.
        self._sized: tuple[torch.Tensor, torch.cuda.Event] | None = None
        self._readback_stream: torch.cuda.Stream | None = None

    def _check(self, endpoint: str, info: DrafterInfo) -> None:
        """Whether this target can use the engine at `endpoint`."""
        spec, mc = self.speculative_config, self.vllm_config.model_config

        def name(model: str) -> str:
            return os.path.basename(model.rstrip("/"))

        why = ""
        if info.num_speculative_tokens != self.num_speculative_steps:
            # Every request has num_speculative_tokens draft slots, whichever
            # engine drafts for it.
            why = (
                f"it drafts {info.num_speculative_tokens} tokens, this target "
                f"verifies {self.num_speculative_steps}"
            )
        elif info.target_model and name(info.target_model) != name(mc.model):
            why = f"its {info.method} head was trained for {info.target_model}"
        elif spec.enable_adaptive_verification and not info.adaptive_verification:
            why = "it was not started with --adaptive-verification"
        elif (
            info.hidden_layers is not None
            and self.vllm_config.cache_config.enable_prefix_caching
        ):
            # On a prefix cache hit the target skips the cached tokens, so the
            # engine never sees their hidden states and the head has no KV for
            # them. A draft model recomputes its KV from the token ids.
            why = (
                f"its {info.method} head cannot follow prefix cache hits; leave "
                "enable_prefix_caching unset (off with drafter_endpoint)"
            )
        if why:
            raise ValueError(f"Drafter engine at {endpoint} cannot be used: {why}.")
        logger.info(
            "Drafter engine at %s: %s %s", endpoint, info.method, name(info.model)
        )

    def set_aux_hidden_state_layers(self, model: nn.Module) -> None:
        """Have the target export the hidden states its engines read."""
        if not supports_eagle3(model):
            raise RuntimeError(
                f"{type(model).__name__} cannot export the hidden states a "
                "drafter head reads"
            )
        target = cast(SupportsEagle3, model)
        default = target.get_eagle3_default_aux_hidden_state_layers()
        # The model returns them in this order.
        self._hidden_layers = self.drafters.use_hidden_layers(default)
        target.set_aux_hidden_state_layers(self._hidden_layers)
        # Part of the hash that names the model's compiled graphs.
        self.speculative_config.drafter_hidden_layers = self._hidden_layers
        logger.info("Exporting hidden states of layers %s", self._hidden_layers)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture(self) -> None:
        pass

    def prepare_watermarking(self, *args, **kwargs) -> None:
        pass

    # A request slot of the model runner is a session of the engines.

    def on_request_added(
        self,
        req_index: int,
        token_ids: list[int],
        num_computed_tokens: int,
        sampling_params: SamplingParams | None,
    ) -> None:
        """A request entered the batch, `num_computed_tokens` into its prompt."""
        args = (sampling_params and sampling_params.extra_args) or {}
        self.drafters.open(req_index, args.get(DRAFTER_ARG))
        self._draft_tokens.pop(req_index, None)
        if (draft_tokens := args.get(DRAFT_TOKENS_ARG)) is not None:
            try:
                self._draft_tokens[req_index] = max(0, int(draft_tokens))
            except (TypeError, ValueError):
                logger.warning_once(
                    "Ignoring %s=%r of a request: not a number",
                    DRAFT_TOKENS_ARG,
                    draft_tokens,
                )
        if num_computed_tokens > 0:
            self._prefixes[req_index] = list(token_ids[:num_computed_tokens])

    def on_request_removed(self, req_index: int) -> None:
        """The model runner freed a request slot: the engine may drop its KV."""
        self._prefixes.pop(req_index, None)
        self._draft_tokens.pop(req_index, None)
        self.drafters.release([req_index])

    def on_step_sized(self, query_start_loc: torch.Tensor) -> None:
        """Adaptive verification has enqueued the sizing of the next step's
        requests; `query_start_loc` will hold the result on the GPU."""
        sized = torch.cuda.Event()
        sized.record()
        self._sized = (query_start_loc, sized)

    def _read_starts(self, num_reqs: int) -> np.ndarray:
        """The exact query_start_loc of a step sized on the GPU. Waits until
        the GPU has sized the step, not for the forward pass enqueued behind
        that: the pass keeps the GPU busy while the CPU cuts the step."""
        assert self._sized is not None
        query_start_loc, sized = self._sized
        if self._readback_stream is None:
            self._readback_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(self._readback_stream):
            self._readback_stream.wait_event(sized)
            return query_start_loc[: num_reqs + 1].cpu().numpy()

    def shutdown(self) -> None:
        self.drafters.detach()

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
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        b, n = input_batch.num_reqs, input_batch.num_tokens
        k = self.num_speculative_steps
        # Warm-up and capture batches hold no request: a send recorded into a
        # CUDA graph would never run, and the engine would wait for it.
        if dummy_run or is_profile or torch.cuda.is_current_stream_capturing():
            return torch.zeros((b, k), dtype=torch.int64, device=self.device)
        if not self.drafters.attached:
            self._attach()

        hidden = None
        if aux_hidden_states:
            hidden = {
                layer: states[:n]
                for layer, states in zip(
                    self._hidden_layers, aux_hidden_states, strict=True
                )
            }
        # With adaptive verification the GPU sizes the requests, and a step
        # that may be cut by engine needs those sizes on the CPU.
        starts = input_batch.query_start_loc_np[: b + 1]
        exact_starts = input_batch.max_query_len is None
        if not exact_starts and self.drafters.num_engines > 1:
            starts, exact_starts = self._read_starts(b), True

        sessions = input_batch.idx_mapping_np[:b].tolist()
        prefixes = {}
        if self._prefixes:
            prefixes = {
                s: self._prefixes.pop(s) for s in sessions if s in self._prefixes
            }
        request = DraftRequest(
            sessions=sessions,
            layout=StepLayout(
                num_tokens=n,
                query_start_loc=starts.tolist(),
                num_scheduled_tokens=input_batch.num_scheduled_tokens[:b].tolist(),
                seq_lens_upper_bound=input_batch.seq_lens_cpu_upper_bound[:b].tolist(),
                is_prefilling=input_batch.is_prefilling_np[:b].tolist(),
                decode_graph_eligible=input_batch.decode_graph_eligible,
                max_query_len=input_batch.max_query_len,
                exact_starts=exact_starts,
            ),
            priority=self.speculative_config.drafter_priority,
            prefixes=prefixes,
            num_draft_tokens=(
                [self._draft_tokens.get(s, k) for s in sessions]
                if self._draft_tokens
                else []
            ),
        )
        idx = input_batch.idx_mapping[:b]
        step = pack_step(
            input_batch.input_ids[:n],
            input_batch.positions[:n],
            input_batch.query_start_loc[: b + 1],
            input_batch.seq_lens[:b],
            num_sampled[:b],
            num_rejected[:b],
            last_sampled.view(-1)[idx],
            next_prefill_tokens[0][idx],
        )
        pending = self.drafters.submit(request, step, hidden)
        drafts, confidences = unpack_reply(self.drafters.collect(pending), k)
        # In batch order, as a local speculator leaves them.
        self.draft_token_confidence_probs[:b] = confidences
        return drafts.to(torch.int64)

    def _attach(self) -> None:
        mc = self.vllm_config.model_config
        scheduler_config = self.vllm_config.scheduler_config
        self.drafters.attach(
            Attach(
                target_model=mc.model,
                num_speculative_tokens=self.num_speculative_steps,
                hidden_size=mc.get_hidden_size(),
                dtype=str(mc.dtype).removeprefix("torch."),
                max_num_seqs=scheduler_config.max_num_seqs,
                max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
                adaptive_verification=self.enable_adaptive_verification,
                label=(
                    f"{os.path.basename(mc.model)}@{socket.gethostname()}:{os.getpid()}"
                ),
            )
        )
