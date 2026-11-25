# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a drafter engine: python -m vllm.v1.drafter.serve --method ... --model ...

Targets reach it with speculative_config = {"method": ..., "model": ...,
"num_speculative_tokens": ..., "drafter_endpoint": "host:port"}.
"""

import argparse
import signal

from vllm.v1.drafter.config import POLICIES, DrafterConfig
from vllm.v1.drafter.engine import DrafterEngine
from vllm.v1.drafter.protocol import METHODS


def parse_args() -> DrafterConfig:
    """Flags are the fields of DrafterConfig, documented there."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--model", required=True, help="draft model or draft head")
    p.add_argument("--target-model", help="the target a head was trained for")
    p.add_argument("--num-speculative-tokens", "-k", type=int, required=True)
    p.add_argument("--adaptive-verification", action="store_true")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--attention-backend")
    p.add_argument("--max-cudagraph-capture-size", type=int)
    p.add_argument(
        "--device",
        default="cuda:0",
        help="the GPU the engine runs on: cuda:<index> among the visible devices",
    )

    p.add_argument(
        "--max-sessions",
        type=int,
        default=256,
        help="sessions in one round; steps that do not fit wait for the next",
    )
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument(
        "--max-live-sessions",
        type=int,
        help="sessions the engine keeps KV for, over all clients "
        "(default: max(4 * --max-sessions, 1024))",
    )
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--kv-cache-reserve-gib", type=float, default=6.0)

    p.add_argument("--host", help="name clients reach this engine by")
    p.add_argument("--port", type=int, default=7700)
    p.add_argument(
        "--scheduling-policy",
        choices=POLICIES,
        default="fcfs",
        help="which waiting steps enter a round first: fcfs = oldest, "
        "priority = lowest drafter_priority of the target, then oldest",
    )
    p.add_argument(
        "--batch-window-ms",
        type=float,
        default=0.0,
        help="how long a step whose payload has landed waits for the "
        "payloads still on their way, to share a round with them",
    )
    p.add_argument(
        "--no-prelaunch",
        dest="prelaunch",
        action="store_false",
        help="always launch a round after its payloads have landed",
    )
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=60.0,
        help="seconds without a request after which a client holding sessions "
        "is taken for dead",
    )
    p.add_argument("--stats-interval", type=float, default=0.0, help="seconds")
    return DrafterConfig(**vars(p.parse_args()))


def main() -> None:
    engine = DrafterEngine(parse_args())

    def stop(*_) -> None:
        engine.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    engine.run()


if __name__ == "__main__":
    main()
