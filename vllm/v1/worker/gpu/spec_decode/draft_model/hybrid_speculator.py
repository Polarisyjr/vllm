# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone draft model with recurrent (GDN) layers on Model Runner V2.

`PlainDraftModelSpeculator` assumes every draft layer is position-addressable:
step 0 replays the target's query (accepted tokens, a correction token and the
rejected slots) and the k-1 decode steps write KV at positions past the
accepted length; stale entries are simply overwritten later. A hybrid drafter
such as Qwen3.5-0.8B also carries GatedDeltaNet layers whose convolution and
recurrent state are *mutable*: feeding rejected drafts or filler into the only
state slot moves it into an unaccepted future, and every later draft round
starts from a wrong prefix.

This speculator keeps the GDN state correct with a zero-copy transactional ring
built from the k+1 Mamba state blocks the scheduler already allocates per
request (`MambaSpec.num_speculative_blocks`, `mamba_cache_mode="none"`). For
each request a committed column `c` is tracked (per request-state slot):

    draft step i (one token):  read (c + i) % N,  write (c + i + 1) % N,  N = k + 1

so the committed slot is never written while drafting. After the target's
verification the ring is reconciled without copying any state:

  * prefill rows replay the chunk in place on the committed slot (their query
    has no drafts);
  * decode rows that accepted `a = m + 1` tokens (the bonus token plus `m`
    drafts, `a <= k`) advance `c += a`: the state after those tokens already
    sits in that ring slot because the drafter consumed them while drafting;
  * decode rows that accepted everything (`a == k + 1`) still have to consume
    the last draft `d_k`, which the drafter proposed but never fed back; one
    extra one-token forward reads slot `(c + k) % N` and writes slot `c`.

Attention layers keep working the position-addressable way: the KV written by
the previous round's draft steps already covers the accepted tokens, so only
the new bonus token and the new drafts are fed, exactly as the ring steps do.

The GDN kernels need separate read and write state indices for this
(`GDNAttentionMetadata.non_spec_state_read_indices_tensor`, the packed
recurrent decode kernel and the cross-block causal-conv update), see the
matching changes in `qwen_gdn_linear_attn.py` and `fused_recurrent.py`.

The k one-token draft steps have a fixed shape (one token per request), so they
are captured as FULL CUDA graphs per batch-size bucket (like the EAGLE/MTP
speculator's decode steps): the ring columns live in persistent buffers that
the captured gathers read, and sampling plus the input/column advance are part
of the graph. The reconciliation forwards (prefill replay, consume-d_k) stay
eager.

Limitations: `mamba_cache_mode="none"` (no prefix caching), data parallel size
1, GDN as the only recurrent layer type.
"""

from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.draft_model.speculator import (
    PlainDraftModelSpeculator,
)
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

_RECURRENT_BACKENDS = {"GDN_ATTN"}


class HybridDraftModelSpeculator(PlainDraftModelSpeculator):
    """Plain draft model whose recurrent layers use a k+1 state ring."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if self.dp_size > 1:
            raise NotImplementedError(
                "HybridDraftModelSpeculator does not support data parallelism."
            )
        self.num_state_slots = self.num_speculative_steps + 1
        # Committed ring column per request-state slot (not per batch row, so
        # that continuous-batching reorders do not move it) and the request id
        # that currently owns the slot, to reset the column on slot reuse.
        self._committed_cols = np.zeros(self.max_num_reqs, dtype=np.int64)
        self._slot_req_ids: list[str | None] = [None] * self.max_num_reqs
        self._state_groups: list[tuple[int, list[str]]] = []
        self._arange_gpu_long = torch.arange(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        # Ring columns of the current draft step, per batch row. Persistent so
        # that captured CUDA graphs can gather the state indices from them.
        self._ring_read_cols = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self._ring_write_cols = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.ring_cudagraph_manager: SpeculatorCudaGraphManager | None = None

    @property
    def attn_vllm_config(self) -> VllmConfig:
        # The GDN builders size their state indices from num_speculative_tokens;
        # the plain speculator's widened view is for attention-only drafters.
        return self.vllm_config

    # ------------------------------------------------------- cuda graphs

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # One-token draft steps only; PIECEWISE is not supported for them.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE
        self.ring_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config, self.device, cudagraph_mode, decode_query_len=1
        )

    def capture(self) -> None:
        if self.ring_cudagraph_manager is None or not self._state_groups:
            return
        logger.info("Capturing CUDA graphs for the hybrid draft model's draft steps...")
        # Reset indices so stale values from dummy runs cannot index out of
        # bounds during capture.
        self.idx_mapping.zero_()
        self._ring_read_cols.zero_()
        self._ring_write_cols.zero_()
        self.current_draft_step.zero_()
        self.ring_cudagraph_manager.capture(
            self._graph_step,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing hybrid draft decode CUDA graphs",
        )

    def _graph_step(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """One ring step on a padded batch; the body of a FULL CUDA graph."""
        assert attn_metadata is not None
        self._override_state_indices(
            attn_metadata,
            num_reqs,
            self._ring_read_cols[:num_reqs],
            self._ring_write_cols[:num_reqs],
            separate_rw=True,
        )
        hidden_states = self._run_model(
            self.input_buffers.input_ids[:num_tokens],
            self.input_buffers.positions[:num_tokens],
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
        )
        self._sample_and_advance(num_reqs, hidden_states)

    # ------------------------------------------------------------ set-up

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
        cache_config = self.vllm_config.cache_config
        if cache_config.mamba_cache_mode != "none":
            raise NotImplementedError(
                "A hybrid draft model requires mamba_cache_mode='none' (disable "
                f"prefix caching); got {cache_config.mamba_cache_mode!r}."
            )
        self._state_groups = []
        self._attention_only_groups: list[list[AttentionGroup]] = []
        recurrent_layers: list[str] = []
        for groups in self.attn_groups:
            self._attention_only_groups.append(
                [g for g in groups if g.backend.get_name() not in _RECURRENT_BACKENDS]
            )
            for attn_group in groups:
                name = attn_group.backend.get_name()
                if name in _RECURRENT_BACKENDS:
                    self._state_groups.append(
                        (attn_group.kv_cache_group_id, list(attn_group.layer_names))
                    )
                    recurrent_layers.extend(attn_group.layer_names)
                elif "MAMBA" in name or "SHORT_CONV" in name or "LINEAR" in name:
                    raise NotImplementedError(
                        f"Recurrent draft layers with backend {name} are not "
                        "supported; only GDN layers are."
                    )
        forward_context = self.vllm_config.compilation_config.static_forward_context
        no_packed = [
            layer_name
            for layer_name in recurrent_layers
            if not getattr(
                forward_context[layer_name], "enable_packed_recurrent_decode", False
            )
        ]
        if no_packed:
            raise NotImplementedError(
                "The hybrid drafter's state ring needs the packed recurrent decode "
                f"kernel (VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1); layers: {no_packed}"
            )
        logger.info(
            "Hybrid draft model %s: %d GDN layer(s) in KV cache group(s) %s use a "
            "%d-slot transactional state ring; %d attention layer(s).",
            self.draft_model_config.model,
            len(recurrent_layers),
            [gid for gid, _ in self._state_groups],
            self.num_state_slots,
            len(self.draft_attn_layer_names) - len(recurrent_layers),
        )

    # ----------------------------------------------------------- helpers

    def _override_state_indices(
        self,
        attn_metadata: dict[str, Any],
        num_rows: int,
        read_cols: torch.Tensor,
        write_cols: torch.Tensor,
        separate_rw: bool,
    ) -> None:
        """Point the GDN layers' metadata at ring columns (per batch row).

        Recurrent layers read their state from `read_cols` and store the
        updated state into `write_cols`; with `separate_rw=False` the slot in
        `write_cols` is updated in place (chunked-prefill path).
        """
        rows = self._arange_gpu_long[:num_rows]
        for group_id, layer_names in self._state_groups:
            # input_block_tables is gathered in the order of the current (sub)batch.
            table = self.block_tables.input_block_tables[group_id]
            write_idx = table[rows, write_cols]
            read_idx = table[rows, read_cols] if separate_rw else None
            for layer_name in layer_names:
                metadata = attn_metadata[layer_name]
                metadata.non_spec_state_indices_tensor = write_idx
                metadata.non_spec_state_read_indices_tensor = read_idx
                if metadata.prefill_state_indices is not None:
                    # Multi-token rows run the chunked prefill path, which reads
                    # and writes one slot in place; mixed batches peel the
                    # one-token rows off the front (see the GDN builder).
                    metadata.prefill_state_indices = write_idx[metadata.num_decodes :]

    def _forward(
        self,
        num_rows: int,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        read_cols: torch.Tensor,
        write_cols: torch.Tensor,
        separate_rw: bool,
        is_prefilling: np.ndarray | None,
    ) -> torch.Tensor:
        """Eagerly run the draft model on rows [0, num_rows) of the input buffers."""
        buffers = self.input_buffers
        num_tokens = int(query_start_loc_np[-1])
        slot_mappings = self.block_tables.compute_slot_mappings(
            self.idx_mapping[:num_rows],
            buffers.query_start_loc[: num_rows + 1],
            buffers.positions,
            num_tokens,
        )
        self.draft_is_prefilling.zero_()
        if is_prefilling is not None:
            self.draft_is_prefilling[:num_rows] = torch.from_numpy(
                np.ascontiguousarray(is_prefilling)
            )
        attn_metadata = self._build_attn_metadata(
            num_reqs=num_rows,
            batch_desc=BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_rows
            ),
            query_start_loc_np=query_start_loc_np,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            slot_mappings=slot_mappings,
        )
        assert attn_metadata is not None
        self._override_state_indices(
            attn_metadata, num_rows, read_cols, write_cols, separate_rw
        )
        hidden_states = self._run_model(
            buffers.input_ids[:num_tokens],
            buffers.positions[:num_tokens],
            attn_metadata,
            build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config),
            None,
        )
        self.draft_is_prefilling.zero_()
        return hidden_states

    def _sample_and_advance(self, num_reqs: int, hidden_states: torch.Tensor) -> None:
        """Sample draft step `current_draft_step` and prepare the next step.

        Device-side only (no Python ints that change per step), so it can be
        part of a captured CUDA graph.
        """
        buffers = self.input_buffers
        draft_tokens = self.sample_draft(
            hidden_states[:num_reqs],
            buffers.positions[:num_reqs],
            self.idx_mapping[:num_reqs],
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        # Clamp so that the warmup/capture passes, which run this more than k
        # times in a row, never index past the last draft column.
        step_index = (
            self.current_draft_step.clamp(max=self.num_speculative_steps - 1)
            .view(1, 1)
            .expand(num_reqs, 1)
        )
        self.draft_tokens[:num_reqs].scatter_(1, step_index, draft_tokens.view(-1, 1))
        buffers.input_ids[:num_reqs].copy_(draft_tokens)
        buffers.positions[:num_reqs].add_(1).clamp_(max=self.max_model_len - 1)
        buffers.seq_lens[:num_reqs].add_(1).clamp_(max=self.max_model_len)
        self._ring_read_cols[:num_reqs].add_(1).remainder_(self.num_state_slots)
        self._ring_write_cols[:num_reqs].add_(1).remainder_(self.num_state_slots)
        self.current_draft_step.add_(1)

    def _run_graph_steps(
        self,
        num_rows: int,
        batch_desc: BatchExecutionDescriptor,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_steps: int,
        first_step: int,
    ) -> None:
        """Replay the captured one-token step `num_steps` times on rows [0, num_rows).

        Before each replay the persistent buffers the captured kernels read are
        refreshed: slot mappings and the attention builders' metadata. The GDN
        layers take their state indices from the ring-column buffers gathered
        inside the graph, so their metadata is not rebuilt.
        """
        assert self.ring_cudagraph_manager is not None
        buffers = self.input_buffers
        query_start_loc_np = self.arange_np[: num_rows + 1]
        all_groups = self.attn_groups
        self.attn_groups = self._attention_only_groups
        try:
            for step in range(num_steps):
                self.block_tables.compute_slot_mappings(
                    self.idx_mapping[:num_rows],
                    buffers.query_start_loc[: num_rows + 1],
                    buffers.positions,
                    batch_desc.num_tokens,
                )
                self._build_attn_metadata(
                    num_reqs=num_rows,
                    batch_desc=batch_desc,
                    query_start_loc_np=query_start_loc_np,
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                    step=first_step + step,
                )
                self.ring_cudagraph_manager.run_fullgraph(batch_desc)
        finally:
            self.attn_groups = all_groups

    def _pad_one_token_batch(self, num_rows: int) -> None:
        """Empty queries and null ring slots for the padded rows of a FULL graph."""
        buffers = self.input_buffers
        buffers.query_start_loc[: num_rows + 1] = self.arange_gpu[: num_rows + 1]
        buffers.query_start_loc[num_rows + 1 :].fill_(num_rows)
        buffers.seq_lens[num_rows:].zero_()
        buffers.input_ids[num_rows : self.max_num_reqs].zero_()
        buffers.positions[num_rows : self.max_num_reqs].zero_()
        self._ring_read_cols[num_rows:].zero_()
        self._ring_write_cols[num_rows:].zero_()

    def _dispatch(self, num_rows: int, is_profile: bool) -> BatchExecutionDescriptor:
        batch_desc, _ = dispatch_cg_and_sync_dp(
            self.ring_cudagraph_manager,
            num_rows,
            num_rows,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile or self.ring_cudagraph_manager is None,
        )
        return batch_desc

    def _select_rows(self, rows_np: np.ndarray, req_indices: torch.Tensor) -> None:
        """Point idx_mapping and the gathered block tables at a subset of rows."""
        rows = torch.as_tensor(rows_np, dtype=torch.long, device=self.device)
        count = int(rows_np.size)
        self.idx_mapping[:count] = req_indices[rows]
        self.block_tables.gather_block_tables(self.idx_mapping[:count], count)

    def _restore_rows(self, num_reqs: int, req_indices: torch.Tensor) -> None:
        self.idx_mapping[:num_reqs] = req_indices
        self.block_tables.gather_block_tables(self.idx_mapping[:num_reqs], num_reqs)

    # ----------------------------------------------------------- propose

    @torch.inference_mode()
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
    ) -> torch.Tensor:
        if dummy_run or not self._state_groups:
            # Dummy block tables point at the null block, whose state the
            # kernels skip or scribble harmlessly; the plain path suffices.
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync=dp_sync,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
            )
        assert self.model is not None
        num_reqs = input_batch.num_reqs
        buffers = self.input_buffers
        num_slots = self.num_state_slots
        device = self.device

        self._copy_request_inputs(num_reqs, input_batch.idx_mapping, temperature, seeds)
        req_indices = input_batch.idx_mapping[:num_reqs]
        req_indices_np = input_batch.idx_mapping_np[:num_reqs]

        # Reset the ring position when a request-state slot changes owner.
        for req_id, slot in zip(input_batch.req_ids, req_indices_np):
            slot = int(slot)
            if self._slot_req_ids[slot] != req_id:
                self._slot_req_ids[slot] = req_id
                self._committed_cols[slot] = 0
        cols_np = self._committed_cols[req_indices_np].copy()

        num_rejected = num_rejected.reshape(-1)[:num_reqs]
        num_rejected_np = num_rejected.cpu().numpy()
        accepted_lens_np = input_batch.num_scheduled_tokens[:num_reqs] - num_rejected_np
        accepted_seq_lens = input_batch.seq_lens[:num_reqs] - num_rejected.int()
        is_prefilling_np = input_batch.is_prefilling_np[:num_reqs].astype(bool)
        qsl_np = input_batch.query_start_loc_np
        seq_upper = input_batch.seq_lens_cpu_upper_bound

        # (1) Prefill rows: replay the chunk in place on the committed slot.
        prefill_rows = np.flatnonzero(is_prefilling_np)
        if prefill_rows.size:
            lens = accepted_lens_np[prefill_rows]
            starts = qsl_np[prefill_rows]
            token_idx = np.concatenate(
                [np.arange(s, s + n) for s, n in zip(starts, lens)]
            )
            query_start_loc_np = np.concatenate(([0], np.cumsum(lens))).astype(np.int32)
            count = int(prefill_rows.size)
            num_tokens = int(query_start_loc_np[-1])
            rows = torch.as_tensor(prefill_rows, dtype=torch.long, device=device)
            token_idx_gpu = torch.as_tensor(token_idx, device=device)
            buffers.input_ids[:num_tokens] = input_batch.input_ids[token_idx_gpu]
            buffers.positions[:num_tokens] = input_batch.positions[token_idx_gpu]
            buffers.query_start_loc[: count + 1] = torch.as_tensor(
                query_start_loc_np, device=device
            )
            buffers.seq_lens[:count] = accepted_seq_lens[rows]
            cols = torch.as_tensor(cols_np[prefill_rows], dtype=torch.long, device=device)
            self._select_rows(prefill_rows, req_indices)
            self._forward(
                count,
                query_start_loc_np,
                seq_upper[torch.from_numpy(prefill_rows)],
                step=0,
                read_cols=cols,
                write_cols=cols,
                separate_rw=False,
                is_prefilling=is_prefilling_np[prefill_rows],
            )
            self._restore_rows(num_reqs, req_indices)

        decode_mask = ~is_prefilling_np
        bad = decode_mask & (
            (accepted_lens_np < 1) | (accepted_lens_np > num_slots)
        )
        if np.any(bad):
            raise RuntimeError(
                "hybrid drafter: decode rows must accept between 1 and "
                f"{num_slots} tokens, got {accepted_lens_np[bad].tolist()}"
            )

        # (2) Fully accepted rows: consume d_k from the last ring slot into the
        # old committed slot, which becomes the committed state again.
        full_rows = np.flatnonzero(decode_mask & (accepted_lens_np == num_slots))
        if full_rows.size:
            count = int(full_rows.size)
            rows = torch.as_tensor(full_rows, dtype=torch.long, device=device)
            last_idx = torch.as_tensor(
                qsl_np[full_rows + 1] - 1, dtype=torch.long, device=device
            )
            buffers.input_ids[:count] = input_batch.input_ids[last_idx]
            buffers.positions[:count] = input_batch.positions[last_idx]
            buffers.seq_lens[:count] = accepted_seq_lens[rows]
            query_start_loc_np = np.arange(count + 1, dtype=np.int32)
            buffers.query_start_loc[: count + 1] = torch.as_tensor(
                query_start_loc_np, device=device
            )
            write_cols = torch.as_tensor(
                cols_np[full_rows], dtype=torch.long, device=device
            )
            read_cols = (write_cols + self.num_speculative_steps) % num_slots
            self._select_rows(full_rows, req_indices)
            batch_desc = self._dispatch(count, is_profile)
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                # Same one-token graph as the draft steps; its sampling and
                # input/column advance are side effects that step (4) resets.
                self._ring_read_cols[:count] = read_cols
                self._ring_write_cols[:count] = write_cols
                self._pad_one_token_batch(count)
                self.block_tables.gather_block_tables(
                    self.idx_mapping[:count], batch_desc.num_reqs or count
                )
                self.current_draft_step.zero_()
                self._run_graph_steps(
                    count, batch_desc, seq_upper[torch.from_numpy(full_rows)], 1, 0
                )
            else:
                self._forward(
                    count,
                    query_start_loc_np,
                    seq_upper[torch.from_numpy(full_rows)],
                    step=0,
                    read_cols=read_cols,
                    write_cols=write_cols,
                    separate_rw=True,
                    is_prefilling=None,
                )
            self._restore_rows(num_reqs, req_indices)

        # (3) Partially accepted rows: the state after the accepted tokens is
        # already in the ring; just move the committed column.
        partial_rows = np.flatnonzero(decode_mask & (accepted_lens_np < num_slots))
        cols_np[partial_rows] = (
            cols_np[partial_rows] + accepted_lens_np[partial_rows]
        ) % num_slots
        self._committed_cols[req_indices_np] = cols_np

        # (4) Draft k tokens, walking the ring from the committed column. The
        # first input is the target's newly sampled token (rows that are still
        # prefilling have no sample; their drafts are discarded by the target).
        buffers.seq_lens[:num_reqs] = accepted_seq_lens
        buffers.input_ids[:num_reqs] = last_sampled.reshape(-1)[req_indices].int()
        buffers.positions[:num_reqs] = accepted_seq_lens.long().clamp_(
            max=self.max_model_len - 1
        )
        buffers.seq_lens[:num_reqs].add_(1).clamp_(max=self.max_model_len)
        query_start_loc_np = np.arange(num_reqs + 1, dtype=np.int32)
        committed_cols = torch.as_tensor(cols_np, dtype=torch.long, device=device)
        self._ring_read_cols[:num_reqs] = committed_cols
        self._ring_write_cols[:num_reqs] = (committed_cols + 1) % num_slots
        self._pad_one_token_batch(num_reqs)
        self.current_draft_step.zero_()

        batch_desc = self._dispatch(num_reqs, is_profile)
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Zero the padded block-table rows so padded rows hit the null slot.
            self.block_tables.gather_block_tables(
                req_indices, batch_desc.num_reqs or num_reqs
            )
            self._run_graph_steps(
                num_reqs, batch_desc, seq_upper, self.num_speculative_steps, 1
            )
        else:
            for step in range(self.num_speculative_steps):
                hidden_states = self._forward(
                    num_reqs,
                    query_start_loc_np,
                    seq_upper,
                    step=step + 1,
                    read_cols=self._ring_read_cols[:num_reqs],
                    write_cols=self._ring_write_cols[:num_reqs],
                    separate_rw=True,
                    is_prefilling=None,
                )
                self._sample_and_advance(num_reqs, hidden_states)

        return self.draft_tokens[:num_reqs]
