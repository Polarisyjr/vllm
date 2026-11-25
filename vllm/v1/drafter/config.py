# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import re
from dataclasses import dataclass

from vllm.v1.drafter.protocol import METHODS

POLICIES = ("fcfs", "priority")


@dataclass
class DrafterConfig:
    """Every knob of a drafter engine. `serve.py` has the same names as flags."""

    # ---- The drafter.
    method: str
    # Drafter checkpoint: a draft model, or a head trained for `target_model`.
    model: str
    num_speculative_tokens: int
    # Heads take the target's embeddings, lm_head and config from here. A
    # draft model needs no target and serves any target with its vocabulary.
    target_model: str | None = None
    # Return acceptance confidences for adaptive verification on the target.
    adaptive_verification: bool = False
    dtype: str = "auto"
    attention_backend: str | None = None
    max_cudagraph_capture_size: int | None = None
    # The GPU the engine runs on: "cuda:<index>" among the devices the
    # process sees.
    device: str = "cuda:0"

    # ---- Capacity.
    # Sessions in one round, as max_num_seqs is the requests in one step of
    # an engine: a step that does not fit waits for the next round.
    max_sessions: int = 256
    # Tokens in one round.
    max_num_batched_tokens: int = 8192
    # Sessions the engine keeps KV for at a time, over all clients (a row
    # each). Default: max(4 * max_sessions, 1024).
    max_live_sessions: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    # Memory left out of the KV cache for CUDA graphs and round buffers.
    kv_cache_reserve_gib: float = 6.0

    # ---- Serving.
    # Name clients reach this engine by (default: the hostname). Control is
    # on `port`, requests on `port + 1`.
    host: str | None = None
    port: int = 7700
    # Which waiting steps enter a round first: "fcfs" = oldest, "priority" =
    # lowest DraftRequest.priority, then oldest.
    scheduling_policy: str = "fcfs"
    # How long a step whose payload has landed waits for the payloads still
    # on their way, to share a round with them. A test knob: an inference
    # engine has no such wait, and neither has this one by default.
    batch_window_ms: float = 0.0
    # Launch the round of a step that is alone in the engine when its request
    # arrives, ahead of its payload (see DrafterEngine.run).
    prelaunch: bool = True
    # Seconds without a request after which a client holding sessions is
    # taken for dead.
    idle_timeout: float = 60.0
    # Seconds between stats lines in the log (0: only at exit).
    stats_interval: float = 0.0

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"drafter method must be one of {METHODS}")
        if self.method != "draft_model" and self.target_model is None:
            raise ValueError(f"a {self.method} head needs --target-model")
        if not re.fullmatch(r"cuda:\d+", self.device):
            raise ValueError(f'device must be "cuda:<index>", not {self.device!r}')
        if self.scheduling_policy not in POLICIES:
            raise ValueError(f"scheduling policy must be one of {POLICIES}")
        if self.max_live_sessions is None:
            self.max_live_sessions = max(4 * self.max_sessions, 1024)
