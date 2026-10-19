# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.utils import next_power_of_2
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    has_compiled_submodule,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup, get_uniform_decode_token_count

logger = init_logger(__name__)


class PlainDraftModelSpeculator(DraftModelSpeculator):
    """Speculative decoding using a separate smaller draft LM.

    Unlike Eagle, the draft model runs fully independently of the target model.
    Step 0 builds an expanded buffer (accepted + correction token + rejected
    slots) via a Triton kernel; steps 1..k-1 are single-token decode steps.

    CUDA graphs. Both phases run from persistent buffers, so they can be
    replayed as CUDA graphs:

    * Step 0 is dispatched through `prefill_cudagraph_manager`. In a pure
      decode batch every request brings 1 bonus token + k drafts and step 0
      appends one correction slot, so the expanded batch is uniform with
      k + 2 tokens per request and replays as a FULL graph per batch-size
      bucket (the padded requests have empty queries and zero sequence
      length). Batches with a prefill run the compiled draft model with
      PIECEWISE graphs when the token count fits a captured size, else eager.
    * Steps 1..k-1 are one token per request and replay as FULL graphs
      through `decode_cudagraph_manager`; the graph body also samples the
      draft token and advances the inputs (token, position, seq_len, step)
      for the next replay, so only the slot mappings and the attention
      metadata are refreshed between steps.
    """

    # Plain draft model needs one extra slot per request for correction.
    num_extra_query_per_req = 1

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # draft_max_seq_len is read by the parent's attention metadata builder;
        # propose() narrows it to the batch like the EAGLE speculator does.
        self.draft_max_seq_len = self.max_model_len

        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        # GPU-side arange for decode query_start_loc initialisation.
        self.arange_gpu = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )
        # Scalar step counter used as column index into draft_logits and
        # draft_tokens; advanced on the device so a graph can carry it.
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)

        _expanded_max = self.max_num_tokens + self.max_num_reqs
        self.expanded_input_ids = torch.zeros(
            _expanded_max, dtype=torch.int32, device=device
        )
        self.expanded_positions = torch.zeros(
            _expanded_max, dtype=torch.int64, device=device
        )
        self.expanded_slot_mappings: torch.Tensor
        self.supports_mm_inputs = False

        # Query width of a decoding request in the expanded step-0 batch:
        # bonus token + k drafts + correction token.
        self.expanded_decode_query_len = self.num_speculative_steps + 2
        # The mode the user configured. The runner later resolves it for the
        # target's verification batches and writes the result back (e.g.
        # PIECEWISE when the backend cannot graph 1 + k token decodes), which
        # says nothing about the drafter's one-token decode steps.
        self.requested_cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        self.prefill_cudagraph_manager: SpeculatorCudaGraphManager | None = None
        self.decode_cudagraph_manager: SpeculatorCudaGraphManager | None = None

    # ------------------------------------------------------------ cuda graphs

    @property
    def attn_vllm_config(self) -> VllmConfig:
        config = copy.copy(super().attn_vllm_config)
        # Spec-as-decode builders (FlashInfer trtllm/XQA) treat a request as a
        # decode only up to 1 + num_speculative_tokens query tokens; the
        # expanded step-0 batch is one wider (correction token), so the draft
        # layers' builders see one more speculative token and keep step 0 on
        # their graph-capturable decode path.
        speculative_config = copy.copy(self.speculative_config)
        speculative_config.num_speculative_tokens = self.num_speculative_steps + 1
        config.speculative_config = speculative_config
        # Builders enable their graph-safe (persistent-buffer) decode path from
        # the compilation config's decode mode. By now the runner has resolved
        # that mode for the target's 1 + k token batches and written it back
        # (PIECEWISE on backends that only graph single-token decodes), so the
        # draft builders are handed the mode the user asked for; the drafter's
        # one-token decode steps decide separately whether to use FULL graphs.
        if self.requested_cudagraph_mode is not None:
            compilation_config = copy.copy(config.compilation_config)
            compilation_config.cudagraph_mode = self.requested_cudagraph_mode
            config.compilation_config = compilation_config
        return config

    def _resolve_cudagraph_mode(
        self, requested: CUDAGraphMode, uniform_query_len: int
    ) -> CUDAGraphMode:
        """`resolve_cudagraph_mode_and_sizes` for the draft layers' backends.

        The runner resolved the mode for the target's verification batches
        (1 + k tokens per request); the drafter's phases have their own widths
        (k + 2 for step 0, 1 for the decode steps), so a backend that only
        graphs single-token decodes still gets FULL decode-step graphs.
        """
        support = self.attn_cg_support.min_cg_support
        piecewise_ok = bool(
            envs.VLLM_USE_BREAKABLE_CUDAGRAPH or has_compiled_submodule(self.model)
        )
        mode = requested
        if (
            mode.mixed_mode() == CUDAGraphMode.FULL
            and support != AttentionCGSupport.ALWAYS
        ):
            mode = (
                CUDAGraphMode.FULL_AND_PIECEWISE
                if piecewise_ok
                else CUDAGraphMode.FULL_DECODE_ONLY
            )
        if mode.decode_mode() == CUDAGraphMode.FULL and (
            support == AttentionCGSupport.NEVER
            or (
                uniform_query_len > 1
                and support.value < AttentionCGSupport.UNIFORM_BATCH.value
            )
        ):
            mode = CUDAGraphMode.PIECEWISE if piecewise_ok else CUDAGraphMode.NONE
        if not piecewise_ok and mode.requires_piecewise_compilation():
            mode = mode.without_piecewise()
        return mode

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        requested = self.requested_cudagraph_mode
        if requested is None or cudagraph_mode == CUDAGraphMode.NONE:
            requested = cudagraph_mode
        # Step 0: FULL graphs for uniform decode batches (k + 2 tokens per
        # request), PIECEWISE through the compiled draft model otherwise.
        prefill_mode = self._resolve_cudagraph_mode(
            requested, self.expanded_decode_query_len
        )
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            prefill_mode,
            decode_query_len=self.expanded_decode_query_len,
        )
        # Steps 1..k-1: one token per request. PIECEWISE is not worth a
        # separate routine here; FULL or nothing, like the EAGLE decode steps.
        if (
            self._resolve_cudagraph_mode(requested, 1).decode_mode()
            == CUDAGraphMode.FULL
        ):
            decode_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            decode_mode = CUDAGraphMode.NONE
        self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config, self.device, decode_mode, decode_query_len=1
        )
        logger.info(
            "Draft model CUDA graphs: step 0 %s, decode steps %s (draft attention "
            "support %s via %s).",
            prefill_mode.name,
            decode_mode.name,
            self.attn_cg_support.min_cg_support.name,
            self.attn_cg_support.min_cg_attn_backend,
        )

    def capture(self) -> None:
        if self.prefill_cudagraph_manager is None:
            return
        logger.info("Capturing CUDA graphs for the draft model...")
        # Reset indices so stale values from dummy runs cannot index out of
        # bounds during capture. idx_mapping == -1 makes sampling skip the row.
        self.last_token_indices.zero_()
        self.idx_mapping.fill_(-1)
        self.current_draft_step.zero_()
        self.expanded_input_ids.zero_()
        self.expanded_positions.zero_()

        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)
        # Step 0 reads its tokens and slot mappings from the expanded buffers,
        # so its capture prepares dummy inputs there (not through the generic
        # speculator helper, which uses the shared slot-mapping buffer).
        CudaGraphManager.capture(
            self.prefill_cudagraph_manager,
            self._create_prefill_capture_fn,
            progress_bar_desc="Capturing draft prefill CUDA graphs",
        )

        if self.num_speculative_steps == 1:
            return
        assert self.decode_cudagraph_manager is not None
        self.decode_cudagraph_manager.capture(
            self._decode_step,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing draft decode CUDA graphs",
        )

    def _create_prefill_capture_fn(
        self, desc: BatchExecutionDescriptor, warmup: bool
    ) -> Callable[[CUDAGraphMode], None]:
        """Dummy step-0 inputs for `desc`, built on the drafter's own buffers."""
        del warmup
        num_tokens = desc.num_tokens
        num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
        full_cudagraph = desc.cg_mode == CUDAGraphMode.FULL
        num_tokens_across_dp = (
            torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
            if self.dp_size > 1
            else None
        )
        max_query_len = desc.max_query_len or desc.uniform_token_count
        if full_cudagraph and max_query_len is None:
            # Mixed graphs can replay a single prefill spanning the batch.
            max_query_len = num_tokens
        input_batch = InputBatch.make_dummy(
            num_reqs, num_tokens, self.input_buffers, max_query_len=max_query_len
        )
        input_block_tables = self.block_tables.get_dummy_block_tables(num_reqs)
        self.expanded_slot_mappings.fill_(PAD_SLOT_ID)
        slot_mappings = self.expanded_slot_mappings[:, :num_tokens]
        if self.block_tables.cp_size > 1:
            input_batch.dcp_local_seq_lens = prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                input_batch.seq_lens,
                input_batch.num_reqs,
                self.block_tables.cp_size,
                self.block_tables.cp_rank,
                self.block_tables.cp_interleave,
                num_reqs_padded=input_batch.num_reqs_after_padding,
            )
        attn_metadata = self.model_state.prepare_attn(
            input_batch,
            CUDAGraphMode.NONE,
            input_block_tables,
            slot_mappings,
            self.attn_groups,
            self.kv_cache_config,
            for_capture=full_cudagraph,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )
        return lambda cg_mode: self._prefill_forward(
            num_reqs,
            num_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp,
            cg_mode,
        )

    # ------------------------------------------------------------------ set-up

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self.expanded_slot_mappings = torch.empty(
            block_tables.num_kv_cache_groups,
            self.max_num_tokens + self.max_num_reqs,
            dtype=torch.int64,
            device=self.device,
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        spec = self.speculative_config
        assert spec is not None
        draft_vllm_config = replace(
            self.vllm_config,
            model_config=self.draft_model_config,
            quant_config=None,
            parallel_config=replace(
                spec.draft_parallel_config,
                rank=self.vllm_config.parallel_config.rank,
            ),
        )
        with set_model_tag("draft_model"):
            return get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )

    # ---------------------------------------------------------------- forward

    @torch.inference_mode()
    def _run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        num_tokens = input_ids.shape[0]
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            model_inputs = dict(input_ids=input_ids, positions=positions)
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                # Compiled piecewise graphs (or breakable graphs), chosen
                # inside run_pw_graph.
                assert self.prefill_cudagraph_manager is not None
                hidden_states = self.prefill_cudagraph_manager.run_pw_graph(
                    self.model, model_inputs
                )
            else:
                hidden_states = self.model(**model_inputs)  # type: ignore[misc]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states

    def _accepted_last_indices(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        qsl = input_batch.query_start_loc
        adjusted_lens = qsl[1 : num_reqs + 1] - qsl[:num_reqs] - num_rejected[:num_reqs]
        return qsl[:num_reqs] + adjusted_lens - 1

    def _sample_first_draft(
        self,
        num_reqs: int,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Draft token 0 from each request's correction-token hidden state."""
        last_indices = self.last_token_indices[:num_reqs]
        self.current_draft_step.zero_()
        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            hidden_states[last_indices],
            positions[last_indices],
            self.idx_mapping[:num_reqs],
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

    def _prefill_forward(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Step 0 on the expanded batch; the body of the prefill CUDA graph.

        Rows [num_reqs_actual, num_reqs) of a FULL graph are padding: their
        last_token_indices are 0 and their idx_mapping is -1, so sampling
        skips them.
        """
        hidden_states = self._run_model(
            self.expanded_input_ids[:num_tokens],
            self.expanded_positions[:num_tokens],
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_first_draft(num_reqs, hidden_states, self.expanded_positions)

    def _prefill(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        if skip_attn:
            # Memory profiling: no KV cache yet, run the target's tokens as is.
            src_tokens = input_batch.num_tokens
            self.last_token_indices[:num_reqs] = self._accepted_last_indices(
                input_batch, num_rejected, num_reqs
            )
            hidden_states = self._run_model(
                input_batch.input_ids[:src_tokens],
                input_batch.positions[:src_tokens],
                None,
                None,
                num_tokens_across_dp,
            )
            self._sample_first_draft(num_reqs, hidden_states, input_batch.positions)
            return

        block_tables = self.block_tables
        kv_cache_config = self.kv_cache_config
        assert block_tables is not None
        assert kv_cache_config is not None

        total_expanded = prepare_prefill_inputs(
            input_buffers=self.input_buffers,
            input_batch=input_batch,
            last_sampled=last_sampled,
            num_rejected=num_rejected,
            expanded_input_ids=self.expanded_input_ids,
            expanded_positions=self.expanded_positions,
            last_token_indices=self.last_token_indices,
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
        )
        # Graph padding (FULL or PIECEWISE) beyond the expanded tokens: keep
        # the tail at valid inputs; its slots are PAD, so nothing is written.
        num_tokens = batch_desc.num_tokens
        assert num_tokens >= total_expanded
        if num_tokens > total_expanded:
            self.expanded_input_ids[total_expanded:num_tokens].zero_()
            self.expanded_positions[total_expanded:num_tokens].zero_()

        prefill_slot_mappings = block_tables.compute_slot_mappings(
            self.idx_mapping[:num_reqs],
            self.input_buffers.query_start_loc[: num_reqs + 1],
            self.expanded_positions[:num_tokens],
            num_tokens,
            out=self.expanded_slot_mappings,
        )
        qsl_np = input_batch.query_start_loc_np
        query_start_loc_np = qsl_np[: num_reqs + 1] + self.arange_np[: num_reqs + 1]
        # The expanded query adds the correction token to every request's
        # sequence, so the bound on the drafter's seq_lens is the target's + 1.
        prefill_attn_md = self._build_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            query_start_loc_np=query_start_loc_np,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            step=1,
            slot_mappings=prefill_slot_mappings,
        )
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # The metadata above refreshed the persistent buffers the graph
            # reads; replay it.
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(batch_desc)
            return
        self._prefill_forward(
            num_reqs,
            num_tokens,
            prefill_attn_md,
            build_slot_mappings_by_layer(prefill_slot_mappings, kv_cache_config),
            num_tokens_across_dp,
            batch_desc.cg_mode,
        )

    def _decode_step(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """One draft decode step on rows [0, num_reqs); the decode graph body.

        Runs one token per request, samples draft token `current_draft_step`
        and advances the inputs (token, position, seq_len, step) for the next
        step. Device-side only, so a captured graph can be replayed k-1 times.
        """
        buffers = self.input_buffers
        hidden_states = self._run_model(
            buffers.input_ids[:num_tokens],
            buffers.positions[:num_tokens],
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        # Clamp so that the warmup/capture passes, which run this more than
        # k - 1 times in a row, never index past the last draft column.
        draft_step = self.current_draft_step.clamp(max=self.num_speculative_steps - 1)
        draft_tokens = self.sample_draft(
            hidden_states[:num_reqs],
            buffers.positions[:num_reqs],
            self.idx_mapping[:num_reqs],
            self.temperature,
            self.seeds,
            draft_step,
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs].scatter_(
            1, draft_step.view(1, 1).expand(num_reqs, 1), draft_tokens.view(-1, 1)
        )
        buffers.input_ids[:num_reqs].copy_(draft_tokens)
        buffers.positions[:num_reqs].add_(1).clamp_(max=self.max_model_len - 1)
        buffers.seq_lens[:num_reqs].add_(1).clamp_(max=self.max_model_len)
        self.current_draft_step.add_(1)

    def _multi_step_decode(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        buffers = self.input_buffers
        if skip_attn:
            last_positions = input_batch.positions[
                self._accepted_last_indices(input_batch, num_rejected, num_reqs)
            ]
        else:
            last_positions = self.expanded_positions[self.last_token_indices[:num_reqs]]

        # Inputs of draft step 1: draft token 0 at the position after the
        # correction token. The drafter's sequence holds the target's accepted
        # tokens, the correction token and this token: target - rejected + 2.
        buffers.input_ids[:num_reqs].copy_(self.draft_tokens[:num_reqs, 0])
        torch.clamp(
            last_positions + 1,
            max=self.max_model_len - 1,
            out=buffers.positions[:num_reqs],
        )
        torch.clamp(
            input_batch.seq_lens[:num_reqs] - num_rejected[:num_reqs].int() + 2,
            max=self.max_model_len,
            out=buffers.seq_lens[:num_reqs],
        )
        buffers.query_start_loc[: num_reqs + 1].copy_(self.arange_gpu[: num_reqs + 1])
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        if num_reqs_padded > num_reqs:
            # Padded rows of a FULL graph: empty queries, zero sequence length
            # and valid (zero) inputs. Their idx_mapping is -1 already.
            buffers.query_start_loc[num_reqs + 1 : num_reqs_padded + 1].fill_(num_reqs)
            buffers.seq_lens[num_reqs:num_reqs_padded].zero_()
            buffers.input_ids[num_reqs:num_reqs_padded].zero_()
            buffers.positions[num_reqs:num_reqs_padded].zero_()
        self.current_draft_step.fill_(1)

        idx_mapping = self.idx_mapping[:num_reqs]
        query_start_loc = buffers.query_start_loc[: num_reqs + 1]
        positions = buffers.positions[:num_reqs]
        for step in range(1, self.num_speculative_steps):
            attn_metadata = None
            slot_mappings_by_layer = None
            if not skip_attn:
                # Refresh the persistent buffers for this step: slot mappings
                # for the new positions and the builders' attention metadata.
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping, query_start_loc, positions, batch_desc.num_tokens
                )
                attn_metadata = self._build_uniform_attn_metadata(
                    batch_desc=batch_desc,
                    num_reqs=num_reqs,
                    num_query_per_req=1,
                    seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                    # Include the correction token inserted during prefill.
                    step=step + 1,
                )
                if batch_desc.cg_mode != CUDAGraphMode.FULL:
                    slot_mappings_by_layer = build_slot_mappings_by_layer(
                        slot_mappings, self.kv_cache_config
                    )
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._decode_step(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp,
                    batch_desc.cg_mode,
                )

    # ---------------------------------------------------------------- propose

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,  # unused
        aux_hidden_states: list[torch.Tensor] | None,  # unused
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
    ) -> torch.Tensor:
        assert self.model is not None

        num_reqs = input_batch.num_reqs
        skip_attn = dummy_run and skip_attn_for_dummy_run
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        max_seq_len = int(seq_lens_cpu_upper_bound[:num_reqs].max())
        self.draft_max_seq_len = min(
            max_seq_len + self.expanded_decode_query_len, self.max_model_len
        )

        # Copy per-request temperature/seeds/idx_mapping into pre-allocated
        # buffers so sample_draft can read them without extra slicing. A dummy
        # batch gets idx_mapping == -1: no KV writes, no sampling.
        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
            dummy_run=dummy_run,
        )

        # Step 0 runs the expanded batch: the target's query plus one
        # correction slot per request. A pure decode batch (every request
        # k + 1 tokens) expands to a uniform k + 2 tokens per request, which
        # is what the FULL prefill graphs were captured for.
        total_expanded = input_batch.num_tokens + num_reqs
        max_query_len = int(input_batch.num_scheduled_tokens[:num_reqs].max())
        uniform_token_count = get_uniform_decode_token_count(
            num_reqs,
            input_batch.num_tokens,
            max_query_len,
            input_batch.decode_graph_eligible,
        )
        if uniform_token_count is not None:
            uniform_token_count += self.num_extra_query_per_req
        prefill_batch_desc, prefill_sync = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            total_expanded,
            uniform_token_count,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=(
                is_profile or skip_attn or self.prefill_cudagraph_manager is None
            ),
        )
        if skip_attn:
            # The profile run forwards the target's tokens as they are.
            num_tokens_across_dp = (
                dp_sync.num_tokens_across_dp if dp_sync is not None else None
            )
        else:
            num_tokens_across_dp = (
                prefill_sync.num_tokens_across_dp if prefill_sync is not None else None
            )

        self._prefill(
            input_batch,
            num_rejected,
            last_sampled,
            num_reqs,
            skip_attn,
            prefill_batch_desc,
            num_tokens_across_dp,
        )

        if self.num_speculative_steps == 1:
            return self.draft_tokens[:num_reqs, :1]

        # Steps 1..k-1: exactly one token per request, so FULL graph replay.
        decode_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync, num_reqs, num_query_per_req=1)
            if dp_sync is not None
            else (None, num_reqs)
        )
        decode_batch_desc, decode_sync = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=(
                is_profile or skip_attn or self.decode_cudagraph_manager is None
            ),
            dp_sync=decode_sync,
        )
        num_tokens_across_dp = (
            decode_sync.num_tokens_across_dp if decode_sync is not None else None
        )
        self._multi_step_decode(
            input_batch,
            num_rejected,
            num_reqs,
            skip_attn,
            decode_batch_desc,
            num_tokens_across_dp,
        )

        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_prefill_inputs_kernel(
    target_input_ids_ptr,  # [src_tokens] int32
    target_positions_ptr,  # [src_tokens] int64
    last_sampled_ptr,  # [max_num_reqs] int32
    idx_mapping_ptr,  # [num_reqs] int32
    out_input_ids_ptr,  # [src_tokens + num_reqs] int32
    out_positions_ptr,  # [src_tokens + num_reqs] int64
    last_token_indices_ptr,  # [max_num_reqs] int64
    out_query_start_loc_ptr,  # [max_num_reqs + 1] int32
    out_seq_lens_ptr,  # [max_num_reqs] int32
    query_start_loc_ptr,  # [num_reqs + 1] int32
    seq_lens_ptr,  # [num_reqs] int32
    num_rejected_ptr,  # [num_reqs] int32
    max_num_reqs,
    max_model_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-request step-0 input preparation for the plain draft-model speculator.

    Output layout for request i (out_start = query_start_loc[i] + i):
        [out_start,              out_start + num_valid)      accepted tokens
        [out_start + num_valid]                              correction token
        (out_start + num_valid, out_start + total_out)      rejected slots

    where num_valid  = query_lens[i] - num_rejected[i]
          total_out  = query_lens[i] + 1
    """
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    q_start = tl.load(query_start_loc_ptr + req_idx)
    q_next = tl.load(query_start_loc_ptr + req_idx + 1)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)

    num_valid = q_next - q_start - num_rejected
    correction_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    start_pos = tl.load(target_positions_ptr + q_start)
    out_start = q_start + req_idx
    total_out = q_next - q_start + 1

    for i in range(0, total_out, BLOCK_SIZE):
        j = i + tl.arange(0, BLOCK_SIZE)
        in_bounds = j < total_out

        is_valid = j < num_valid
        is_correction = j == num_valid
        src_idx = q_start + j
        token_ids = tl.load(target_input_ids_ptr + src_idx, mask=is_valid, other=0)
        positions = tl.minimum(start_pos + j, max_model_len - 1)
        token_ids = tl.where(is_correction, correction_token, token_ids)

        out_idx = out_start + j
        tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
        tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)

    tl.store(last_token_indices_ptr + req_idx, out_start + num_valid)
    tl.store(out_query_start_loc_ptr + req_idx, out_start)
    # seqlen_k for the expanded prefill attention:
    #   pre_existing = seq_len - query_len (draft KV before this round)
    #   expanded_query_len = query_len + 1 (adds correction token)
    #   seqlen_k = pre_existing + expanded_query_len = seq_len + 1
    # num_rejected does NOT change seqlen_k — the expanded query always has
    # query_len+1 slots. Rejected positions write temporary KV entries that
    # are overwritten with valid tokens in a later round.
    new_seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(out_seq_lens_ptr + req_idx, new_seq_len)

    if req_idx == num_reqs - 1:
        # Pad for CUDA graphs: empty queries, zero seq_lens and a valid
        # last-token index for the rows a FULL graph adds.
        total_expanded = out_start + total_out
        tl.store(out_query_start_loc_ptr + num_reqs, total_expanded)
        for i in range(num_reqs + 1, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block <= max_num_reqs
            tl.store(out_query_start_loc_ptr + block, total_expanded, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_prefill_inputs(
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    last_sampled: torch.Tensor,  # [max_num_reqs] int64
    num_rejected: torch.Tensor,  # [num_reqs]     int64
    expanded_input_ids: torch.Tensor,  # [max_tokens + max_reqs] int32
    expanded_positions: torch.Tensor,  # [max_tokens + max_reqs] int64
    last_token_indices: torch.Tensor,  # [max_num_reqs] int64
    max_num_reqs: int,
    max_model_len: int,
) -> int:
    """Call _prepare_prefill_inputs_kernel and return total_expanded tokens.

    Side-effects (kernel writes):
      - expanded_input_ids, expanded_positions
      - last_token_indices
      - input_buffers.query_start_loc  (expanded: original[i] + i)
      - input_buffers.seq_lens         (= target_seq_lens + 1)
    """
    num_reqs = input_batch.num_reqs
    src_tokens = input_batch.num_tokens
    qsl_np = input_batch.query_start_loc_np
    query_lens = qsl_np[1 : num_reqs + 1] - qsl_np[:num_reqs]
    max_total_out = int(query_lens.max()) + 2  # +1 correction, +1 alignment
    BLOCK_SIZE = min(512, next_power_of_2(max_total_out))

    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_batch.input_ids,
        input_batch.positions,
        last_sampled.int(),
        input_batch.idx_mapping,
        expanded_input_ids,
        expanded_positions,
        last_token_indices,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        num_rejected.int(),
        max_num_reqs,
        max_model_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return src_tokens + num_reqs
