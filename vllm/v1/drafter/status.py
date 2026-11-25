# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Occupancy of a drafter engine: python -m vllm.v1.drafter.status host:port

With --draft-tokens N the engine drafts at most N tokens per session from its
next round on."""

import argparse
import dataclasses
import json

from vllm.v1.drafter.client import query_status, set_draft_tokens

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("endpoint", help="host:port")
    p.add_argument("--draft-tokens", type=int)
    args = p.parse_args()
    if args.draft_tokens is None:
        status = query_status(args.endpoint)
    else:
        status = set_draft_tokens(args.endpoint, args.draft_tokens)
    print(json.dumps(dataclasses.asdict(status), indent=2))
