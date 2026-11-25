# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GPU half of the drafter engine: one stock speculator, no target model.

`DraftRunner` does for a speculator what the model runner does for it inside
a target engine: it owns the KV cache, the block tables and the input
buffers, and hands `propose()` an `InputBatch` that describes a target step.
The step comes from the clients instead of a local forward pass, and a round
may hold the steps of several clients back to back. The speculators
themselves (standalone draft model, EAGLE-3 head, DFlash head) run unchanged.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open

from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    graph_capture,
    init_distributed_environment,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.utils.gc_utils import freeze_gc_for_cudagraph_capture
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import async_tensor_h2d, set_random_seed
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    record_kv_cache_layout,
    resolve_kv_cache_layout,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.drafter.config import DrafterConfig
from vllm.v1.drafter.protocol import StepLayout, packed_step_numel, unpack_step
from vllm.v1.drafter.sessions import OVERFLOW_ROW, SessionStore
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import has_compiled_submodule
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)
from vllm.v1.worker.utils import get_uniform_decode_token_count

logger = init_logger(__name__)

GiB = 1 << 30


@dataclass
class StepInput:
    """One client's step as a round takes it."""

    # The row of each session of the step (see SessionStore).
    rows: list[int]
    layout: StepLayout
    # protocol.pack_step, and [num_tokens, num_hidden] for a head.
    step: torch.Tensor
    hidden: torch.Tensor | None = None
    # Tokens to draft per session, at most. Empty: as many as the round may.
    num_draft_tokens: list[int] = field(default_factory=list)


@dataclass
class _RoundLayout:
    """The steps of a round back to back, as the CPU knows them."""

    num_sessions: int
    num_tokens: int
    rows: np.ndarray
    query_start_loc: np.ndarray
    num_scheduled_tokens: np.ndarray
    seq_lens_upper_bound: np.ndarray
    is_prefilling: np.ndarray
    # Tokens to draft per session.
    draft_tokens: np.ndarray
    decode_graph_eligible: bool
    # A client sizes its steps on the GPU (adaptive verification).
    varlen: bool


class _TargetStub(nn.Module):
    """What a head loads from its target: the embeddings and the lm_head."""

    def __init__(self, vllm_config: VllmConfig, with_weights: bool):
        super().__init__()
        mc = vllm_config.model_config
        self.config = mc.hf_config
        self.model = nn.Module()
        if not with_weights:
            return
        vocab, hidden = mc.get_vocab_size(), mc.get_hidden_size()
        self.model.embed_tokens = VocabParallelEmbedding(
            vocab, hidden, params_dtype=mc.dtype, prefix="model.embed_tokens"
        )
        self.lm_head = ParallelLMHead(
            vocab, hidden, params_dtype=mc.dtype, prefix="lm_head"
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def load_weights(self, model_path: str) -> None:
        root = Path(model_path)
        index = root / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
        else:
            weight_map = {}
            for f in sorted(root.glob("*.safetensors")):
                with safe_open(str(f), "pt", device="cpu") as sf:
                    weight_map.update(dict.fromkeys(sf.keys(), f.name))

        def read(name: str) -> torch.Tensor | None:
            if name not in weight_map:
                return None
            with safe_open(str(root / weight_map[name]), "pt", device="cpu") as sf:
                return sf.get_tensor(name)

        embed = read("model.embed_tokens.weight")
        if embed is None:
            raise ValueError(f"{model_path} has no model.embed_tokens.weight")
        self.model.embed_tokens.weight_loader(self.model.embed_tokens.weight, embed)
        lm_head = read("lm_head.weight")
        if lm_head is None:
            assert getattr(self.config, "tie_word_embeddings", False)
            lm_head = embed
        self.lm_head.weight_loader(self.lm_head.weight, lm_head)


def pad_drafts(
    drafts: torch.Tensor,
    confidences: torch.Tensor,
    draft_tokens: torch.Tensor,
    cols: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The first `draft_tokens[i]` drafts of session i and padding behind
    them, [num_sessions, k]. `drafts` may be narrower than k, `cols` is
    arange(k)."""
    missing = cols.shape[0] - drafts.shape[1]
    if missing > 0:
        drafts = torch.nn.functional.pad(drafts, (0, missing))
    drafted = cols[None, :] < draft_tokens[:, None]
    # Any token of the vocabulary will do, -1 would not: should the target
    # verify padding, it runs it through its forward pass.
    return torch.where(drafted, drafts, 0), torch.where(drafted, confidences, 0.0)


def clip_verdict(
    num_sampled: torch.Tensor, num_rejected: torch.Tensor, drafted: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The target's verdict on the `drafted` tokens of each session, without
    the padding it may have verified behind them: the drafter predicted
    nothing there to be graded on."""
    accepted = (num_sampled - 1).clamp(min=0)
    return (
        torch.minimum(num_sampled, drafted + 1),
        torch.minimum(num_rejected, (drafted - accepted).clamp(min=0)),
    )


class DraftRunner:
    """Interface: `sessions` to open, grow and close sessions, `propose()` to
    draft for a round of steps, `prefill()` to give sessions KV for tokens no
    step will bring."""

    def __init__(self, config: DrafterConfig):
        self.config = config
        self.method = config.method
        self.k = config.num_speculative_tokens
        self.device = torch.device(config.device)
        assert config.max_live_sessions is not None
        # A row per live session, and the overflow row.
        self.max_rows = config.max_live_sessions + 1
        self.vllm_config = self._engine_config()
        with set_current_vllm_config(self.vllm_config), torch.inference_mode():
            stub = self._load_speculator()
            cg_support = self._init_kv_cache(stub)
            cudagraph_mode = self._capture_graphs(cg_support)
            self._init_state()
        logger.info(
            "Drafter ready: %s %s, k=%d, %d sessions per round, %d KV blocks x %d, "
            "cudagraph mode %s",
            self.method,
            config.model,
            self.k,
            config.max_sessions,
            self.sessions.num_blocks,
            self.sessions.block_size,
            cudagraph_mode,
        )

    # --------------------------------------------------------------- setup

    def _engine_config(self) -> VllmConfig:
        """The config of a target engine that would host this speculator."""
        config = self.config
        spec: dict = {
            "method": config.method,
            "model": config.model,
            "num_speculative_tokens": self.k,
        }
        if config.adaptive_verification:
            spec["enable_adaptive_verification"] = True
        extra: dict = {}
        if config.attention_backend:
            extra["attention_backend"] = config.attention_backend
        if config.max_cudagraph_capture_size is not None:
            extra["max_cudagraph_capture_size"] = config.max_cudagraph_capture_size
        return EngineArgs(
            model=config.target_model or config.model,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            max_num_seqs=self.max_rows,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=False,
            speculative_config=spec,
            **extra,
        ).create_engine_config()

    def _load_speculator(self) -> _TargetStub:
        cfg, config = self.vllm_config, self.config
        torch.cuda.set_device(self.device)
        init_distributed_environment(
            world_size=1,
            rank=0,
            # The device the process groups take for theirs.
            local_rank=self.device.index,
            backend="nccl",
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        ensure_model_parallel_initialized(1, 1)
        set_random_seed(cfg.model_config.seed)

        is_head = self.method != "draft_model"
        stub = _TargetStub(cfg, with_weights=is_head).to(self.device)
        if is_head:
            assert config.target_model is not None
            stub.load_weights(config.target_model)
        self.speculator = init_speculator(cfg, self.device)
        self.speculator.load_model(stub)
        # The target's layers whose hidden states a head reads, as a target
        # that hosts it would pick them: the ones its checkpoint names, or
        # (empty) the target model's default ones.
        self.hidden_layers: tuple[int, ...] | None = None
        if is_head:
            layers = get_eagle3_aux_layers_from_config(cfg.speculative_config)
            self.hidden_layers = tuple(sorted(layers or ()))
        # A target-dependent drafter runs its first pass over the target's
        # batch, with the attention metadata of that batch.
        self._replays_target_batch = isinstance(
            self.speculator, TargetDependentARSpeculator
        )
        self.target_hidden_size = cfg.model_config.get_hidden_size()
        self.dtype = cfg.model_config.dtype
        self.max_num_tokens = cfg.scheduler_config.max_num_batched_tokens
        self.max_model_len = cfg.model_config.max_model_len
        return stub

    def _init_kv_cache(self, stub: _TargetStub) -> Any:
        cfg, config = self.vllm_config, self.config
        specs = get_kv_cache_spec(cfg)
        assert specs, "the drafter has no KV cache layers"
        # What EngineCore does once for all workers: pick the KV layout the
        # backends support and record it in the cache config.
        backends = get_current_attn_backends(cfg)
        layout = resolve_kv_cache_layout(
            cfg,
            [[x.name for x in get_supported_kv_cache_layouts(backends)]],
            list(specs.values()),
        )
        record_kv_cache_layout(cfg.cache_config, layout.name)
        free, total = torch.cuda.mem_get_info(self.device)
        available = (
            int(total * config.gpu_memory_utilization)
            - (total - free)
            - int(config.kv_cache_reserve_gib * GiB)
        )
        if available <= 0:
            raise RuntimeError("no GPU memory left for the drafter KV cache")
        self.kv_cache_config = get_kv_cache_configs(cfg, [specs], [available])[0]
        if len(self.kv_cache_config.kv_cache_groups) != 1:
            raise NotImplementedError("drafters with several KV cache groups")
        self.attn_groups, cg_support, kernel_block_sizes = init_attn_backend(
            self.kv_cache_config, cfg, self.device
        )
        block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        # A session writes its step and then up to k + 1 draft positions.
        self.session_headroom = self.k + 2
        self.max_blocks_per_session = cdiv(
            self.max_model_len + self.session_headroom, block_size
        )
        self.block_tables = BlockTables(
            block_sizes=[block_size],
            max_num_reqs=self.max_rows,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=[self.max_blocks_per_session],
            device=self.device,
            kernel_block_sizes=kernel_block_sizes,
        )
        self.input_buffers = InputBuffers(
            self.max_rows, self.max_num_tokens, self.device
        )
        model_state = DefaultModelState(cfg, stub, None, self.device)
        self.kv_caches = init_kv_cache(
            cfg.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.device,
            kernel_block_sizes,
            cfg,
            block_tables=self.block_tables,
        )
        self.speculator.set_attn(
            model_state,
            self.kv_cache_config,
            self.block_tables,
            self.input_buffers,
            self.attn_groups,
        )
        return cg_support

    def _capture_graphs(self, cg_support: Any) -> CUDAGraphMode:
        cfg = self.vllm_config
        cudagraph_mode = cfg.compilation_config.resolve_cudagraph_mode_and_sizes(
            cg_support.min_cg_support,
            cg_support.min_cg_attn_backend,
            self.k + 1,
            use_v2_model_runner=True,
            tensor_parallel_size=1,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_rows,
            is_profiling=False,
            piecewise_capture_available=has_compiled_submodule(self.speculator.model),
        )
        self.speculator.init_cudagraph_manager(cudagraph_mode)
        # A drafter that fuses its draft steps into one graph needs one per
        # number of steps a round may run. The speculator captures the numbers
        # that this table names, whatever batch sizes it names them for; it is
        # set after the managers exist so that it shapes nothing else.
        spec = cfg.speculative_config
        assert spec is not None
        spec.num_speculative_tokens_per_batch_size = [
            (i, i, i) for i in range(1, self.k + 1)
        ]
        set_cudagraph_capturing_enabled(True)
        with freeze_gc_for_cudagraph_capture(), graph_capture(device=self.device):
            self.speculator.capture()
        set_cudagraph_capturing_enabled(False)
        spec.num_speculative_tokens_per_batch_size = None
        torch.cuda.synchronize()
        return cudagraph_mode

    def _init_state(self) -> None:
        self.sessions = SessionStore(
            self.block_tables,
            self.kv_cache_config.num_blocks,
            self.max_blocks_per_session,
            self.session_headroom,
        )
        # Per-row state the speculators index by row.
        n, dev = self.max_rows, self.device
        self.last_sampled = torch.zeros(n, 1, dtype=torch.int64, device=dev)
        self.next_prefill = torch.zeros(1, n, dtype=torch.int32, device=dev)
        self.temperature = torch.zeros(n, dtype=torch.float32, device=dev)
        self.seeds = torch.zeros(n, dtype=torch.int64, device=dev)
        # Per-round buffers.
        self.num_sampled = torch.zeros(n, dtype=torch.int32, device=dev)
        self.num_rejected = torch.zeros(n, dtype=torch.int32, device=dev)
        self.hidden: torch.Tensor | None = None
        # Tokens drafted for each row in its last round (the rest of its k were
        # padding), kept once a round has drafted fewer than k for a session.
        self.drafted = torch.full((n,), self.k, dtype=torch.int32, device=dev)
        self._drafts_vary = False
        # Draft tokens the last round ran for.
        self.round_draft_tokens = self.k
        self._draft_cols = torch.arange(self.k, dtype=torch.int32, device=dev)
        self._ones = torch.ones(n, self.k, dtype=torch.float32, device=dev)
        self._zeros_i32 = torch.zeros(n, dtype=torch.int32, device=dev)
        self._no_padding = torch.zeros(
            self.max_num_tokens, dtype=torch.bool, device=dev
        )
        self._arange = torch.arange(n + 1, dtype=torch.int32, device=dev)
        self._arange_np = np.arange(n + 1, dtype=np.int32)

    # ------------------------------------------------------------- propose

    @torch.inference_mode()
    def propose(
        self, steps: list[StepInput], max_draft_tokens: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft for the sessions of `steps`, in order, up to
        `max_draft_tokens` tokens each.

        Every row must be reserved (`sessions.reserve`) for its session's
        length after the step. Returns [num_sessions, k] draft tokens and
        acceptance confidences (ones unless the drafter estimates them). Both
        are views of buffers the next round overwrites. Nothing here waits
        for the GPU: the engine may launch a round before its payload has
        landed.

        A round runs as many draft steps as the session that gets the most
        tokens needs (DFlash drafts all k in one pass regardless). Past its
        own number a session gets padding: token 0, confidence 0.
        """
        with set_current_vllm_config(self.vllm_config):
            return self._propose(
                steps, self.k if max_draft_tokens is None else max_draft_tokens
            )

    def _propose(
        self, steps: list[StepInput], max_draft_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ib = self.input_buffers
        cpu = self._merge(steps, max_draft_tokens)
        b, n = cpu.num_sessions, cpu.num_tokens
        self.block_tables.apply_staged_writes()
        idx_mapping = async_tensor_h2d(cpu.rows, device=self.device, dtype=torch.int32)

        has_prefill = bool(cpu.is_prefilling.any())
        max_query_len = int(cpu.num_scheduled_tokens.max())
        nr, nt = b, n
        if self._replays_target_batch:
            # The first pass replays a FULL graph captured over these input
            # buffers, so the round is staged at the graph's padded shape, as
            # the model runner stages a target batch.
            uniform = get_uniform_decode_token_count(
                b, n, max_query_len, cpu.decode_graph_eligible
            )
            desc, _ = dispatch_cg_and_sync_dp(
                self.speculator.prefill_cudagraph_manager,
                b,
                n,
                uniform,
                dp_size=1,
                dp_rank=0,
                max_query_len=None if has_prefill else max_query_len,
            )
            if desc.cg_mode == CUDAGraphMode.FULL:
                nr, nt = desc.num_reqs or b, desc.num_tokens

        hidden = self._stage(steps, idx_mapping, b, n, nr, nt)
        attn_metadata: dict = {}
        slot_mappings_by_layer: dict = {}
        if self._replays_target_batch:
            attn_metadata, slot_mappings_by_layer = self._target_attn_inputs(
                cpu, idx_mapping, nr, nt, max_query_len
            )

        num_computed = cpu.seq_lens_upper_bound - cpu.num_scheduled_tokens
        input_batch = InputBatch(
            req_ids=[],
            num_reqs=b,
            num_reqs_after_padding=nr,
            idx_mapping=idx_mapping,
            idx_mapping_np=cpu.rows,
            expanded_idx_mapping=idx_mapping,
            expanded_local_pos=self._zeros_i32[:b],
            num_scheduled_tokens=cpu.num_scheduled_tokens,
            num_tokens=n,
            num_tokens_after_padding=nt,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=ib.query_start_loc[: b + 1],
            query_start_loc_np=cpu.query_start_loc,
            seq_lens=ib.seq_lens[:nr],
            seq_lens_cpu_upper_bound=torch.from_numpy(cpu.seq_lens_upper_bound),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=num_computed,
            prefill_len_np=np.where(
                cpu.is_prefilling, cpu.seq_lens_upper_bound + 1, num_computed
            ),
            num_computed_prefill_tokens_np=num_computed,
            is_prefilling_np=cpu.is_prefilling,
            has_prefill=has_prefill,
            decode_graph_eligible=cpu.decode_graph_eligible,
            input_ids=ib.input_ids[:nt],
            positions=ib.positions[:nt],
            is_padding=self._no_padding[:nt],
            logits_indices=ib.query_start_loc[1 : b + 1] - 1,
            cu_num_logits=self._arange[: b + 1],
            cu_num_logits_np=self._arange_np[: b + 1],
            has_structured_output_reqs=False,
            prompt_lens=None,
            max_query_len=max_query_len if cpu.varlen else None,
        )

        last_hidden = aux = None
        if hidden is not None:
            hidden = hidden[:nt]
            if hidden.shape[-1] > self.target_hidden_size:
                aux = list(torch.split(hidden, self.target_hidden_size, dim=-1))
                last_hidden = aux[-1]
            else:
                last_hidden = hidden
        num_sampled, num_rejected = self.num_sampled[:b], self.num_rejected[:b]
        self.round_draft_tokens = int(cpu.draft_tokens.max())
        # Once, not only in such a round: the next one grades its padding.
        self._drafts_vary |= bool((cpu.draft_tokens < self.k).any())
        if self._drafts_vary:
            verdict = clip_verdict(num_sampled, num_rejected, self.drafted[idx_mapping])
        else:
            verdict = num_sampled, num_rejected
        self.speculator.observe_verification(idx_mapping, *verdict)
        drafts = self.speculator.propose(
            input_batch,
            attn_metadata,
            slot_mappings_by_layer,
            last_hidden,
            aux,
            num_sampled,
            num_rejected,
            self.last_sampled,
            self.next_prefill,
            self.temperature,
            self.seeds,
            num_speculative_tokens=self.round_draft_tokens,
        )
        if self.speculator.acceptance_estimator is not None:
            # In batch order, like the drafts.
            confidences = self.speculator.draft_token_confidence_probs[:b]
        else:
            confidences = self._ones[:b]
        drafts = drafts[:b]
        if self._drafts_vary:
            draft_tokens = async_tensor_h2d(
                cpu.draft_tokens, device=self.device, dtype=torch.int32
            )
            self.drafted[idx_mapping] = draft_tokens
            drafts, confidences = pad_drafts(
                drafts, confidences, draft_tokens, self._draft_cols
            )
        return drafts, confidences

    def _merge(self, steps: list[StepInput], max_draft_tokens: int) -> _RoundLayout:
        b = sum(len(s.rows) for s in steps)
        n = sum(s.layout.num_tokens for s in steps)
        query_start_loc = np.empty(b + 1, dtype=np.int32)
        r = t = 0
        for s in steps:
            query_start_loc[r : r + len(s.rows)] = (
                np.asarray(s.layout.query_start_loc[:-1]) + t
            )
            r += len(s.rows)
            t += s.layout.num_tokens
        query_start_loc[b] = n
        layouts = [s.layout for s in steps]
        return _RoundLayout(
            num_sessions=b,
            num_tokens=n,
            rows=np.fromiter((x for s in steps for x in s.rows), np.intp, b),
            query_start_loc=query_start_loc,
            num_scheduled_tokens=np.fromiter(
                (x for y in layouts for x in y.num_scheduled_tokens), np.int32, b
            ),
            seq_lens_upper_bound=np.fromiter(
                (x for y in layouts for x in y.seq_lens_upper_bound), np.int32, b
            ),
            is_prefilling=np.fromiter(
                (x for y in layouts for x in y.is_prefilling), np.bool_, b
            ),
            draft_tokens=np.fromiter(
                (
                    min(x, max_draft_tokens)
                    for s in steps
                    for x in s.num_draft_tokens or [max_draft_tokens] * len(s.rows)
                ),
                np.int32,
                b,
            ),
            decode_graph_eligible=all(y.decode_graph_eligible for y in layouts),
            varlen=any(y.max_query_len is not None for y in layouts),
        )

    def _stage(
        self,
        steps: list[StepInput],
        idx_mapping: torch.Tensor,
        b: int,
        n: int,
        nr: int,
        nt: int,
    ) -> torch.Tensor | None:
        """Copy the payloads into the input buffers and the per-row state,
        padded to [nr] sessions and [nt] tokens. Returns the hidden states."""
        ib = self.input_buffers
        hidden = None
        if steps[0].hidden is not None:
            num_hidden = steps[0].hidden.shape[-1]
            if self.hidden is None or self.hidden.shape[-1] != num_hidden:
                self.hidden = torch.zeros(
                    (self.max_num_tokens, num_hidden),
                    dtype=self.dtype,
                    device=self.device,
                )
            hidden = self.hidden
        r = t = 0
        for s in steps:
            bj, nj = len(s.rows), s.layout.num_tokens
            f = unpack_step(s.step, bj, nj)
            ib.input_ids[t : t + nj].copy_(f["input_ids"])
            ib.positions[t : t + nj].copy_(f["positions"])
            torch.add(f["query_start_loc"][:-1], t, out=ib.query_start_loc[r : r + bj])
            ib.seq_lens[r : r + bj].copy_(f["seq_lens"])
            self.num_sampled[r : r + bj].copy_(f["num_sampled"])
            self.num_rejected[r : r + bj].copy_(f["num_rejected"])
            rows = idx_mapping[r : r + bj]
            self.last_sampled[rows, 0] = f["last_sampled"].to(torch.int64)
            self.next_prefill[0, rows] = f["next_prefill"]
            if hidden is not None:
                hidden[t : t + nj].copy_(s.hidden)
            r += bj
            t += nj
        ib.query_start_loc[b : nr + 1].fill_(n)
        ib.seq_lens[b:nr].zero_()
        ib.positions[n:nt].zero_()
        ib.input_ids[n:nt].zero_()
        self.block_tables.gather_block_tables(idx_mapping, num_reqs_padded=nr)
        return hidden

    def _target_attn_inputs(
        self,
        cpu: _RoundLayout,
        idx_mapping: torch.Tensor,
        nr: int,
        nt: int,
        max_query_len: int,
    ) -> tuple[dict, dict]:
        """Attention metadata and slot mappings of the round as a target
        batch, for a drafter whose first pass runs over that batch."""
        ib = self.input_buffers
        b, n = cpu.num_sessions, cpu.num_tokens
        slot_mappings = self.block_tables.compute_slot_mappings(
            idx_mapping, ib.query_start_loc[: b + 1], ib.positions[:nt], nt
        )
        qsl_pad = np.full(nr + 1, n, dtype=np.int32)
        qsl_pad[: b + 1] = cpu.query_start_loc
        seq_pad = np.zeros(nr, dtype=np.int32)
        seq_pad[:b] = cpu.seq_lens_upper_bound
        prefill_pad = np.zeros(nr, dtype=np.bool_)
        prefill_pad[:b] = cpu.is_prefilling
        attn_metadata = build_attn_metadata(
            self.attn_groups,
            nr,
            nt,
            ib.query_start_loc[: nr + 1],
            torch.from_numpy(qsl_pad),
            max_query_len,
            ib.seq_lens[:nr],
            int(cpu.seq_lens_upper_bound.max()),
            [x[:nr] for x in self.block_tables.input_block_tables],
            slot_mappings,
            self.kv_cache_config,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_pad),
            positions=ib.positions,
            is_prefilling=torch.from_numpy(prefill_pad),
        )
        return attn_metadata, build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )

    # ------------------------------------------------------------- prefill

    def prefill(self, rows: list[int], prefixes: list[list[int]]) -> int:
        """Compute the KV of `prefixes[i]`, the first tokens of the session at
        `rows[i]`, as prompt chunks. Returns the number of tokens.

        Only a draft model can: a head needs the target's hidden states. The
        rows must be reserved for the prefixes. Runs as rounds of its own and
        blocks on a host-to-device copy, outside the engine's scheduling.
        """
        assert self.method == "draft_model"
        todo = [
            (row, tokens, 0)
            for row, tokens in zip(rows, prefixes)
            if row != OVERFLOW_ROW
        ]
        num_tokens = 0
        while todo:
            # One chunk per session and round, within the token budget (a
            # draft model adds a slot per session to its first pass).
            budget = self.max_num_tokens - len(todo)
            share = max(1, budget // len(todo))
            chunk_rows, chunks, starts, rest = [], [], [], []
            for row, tokens, start in todo:
                end = min(len(tokens), start + share)
                chunk_rows.append(row)
                chunks.append(tokens[start:end])
                starts.append(start)
                if end < len(tokens):
                    rest.append((row, tokens, end))
            todo = rest
            lens = [len(c) for c in chunks]
            b, n = len(chunk_rows), sum(lens)
            query_start_loc = np.zeros(b + 1, dtype=np.int32)
            np.cumsum(lens, out=query_start_loc[1:])
            seq_lens = [s + x for s, x in zip(starts, lens)]
            step = torch.zeros(packed_step_numel(b, n), dtype=torch.int32)
            f = unpack_step(step, b, n)
            f["input_ids"].copy_(
                torch.tensor([t for c in chunks for t in c], dtype=torch.int32)
            )
            f["positions"].copy_(
                torch.cat(
                    [
                        torch.arange(s, e, dtype=torch.int32)
                        for s, e in zip(starts, seq_lens)
                    ]
                )
            )
            f["query_start_loc"].copy_(torch.from_numpy(query_start_loc))
            f["seq_lens"].copy_(torch.tensor(seq_lens, dtype=torch.int32))
            layout = StepLayout(
                num_tokens=n,
                query_start_loc=query_start_loc.tolist(),
                num_scheduled_tokens=lens,
                seq_lens_upper_bound=seq_lens,
                is_prefilling=[True] * b,
            )
            self.propose([StepInput(chunk_rows, layout, step.to(self.device))])
            num_tokens += n
        return num_tokens
