# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafter engine: a drafter served from its own process and GPU.

The engine owns one drafter (a draft model or a draft head), its KV cache and
its CUDA graphs, and drafts for any number of target engines. See README.md
for the vocabulary and the life of a step.

Shared by both sides:
  protocol.py   messages and tensor layouts on the wire
  transport.py  the NCCL link of a client

Engine side (`python -m vllm.v1.drafter.serve`):
  config.py     DrafterConfig, every knob of an engine
  engine.py     DrafterEngine: the loop (clients in, rounds out)
  links.py      Client: an attached target and its NCCL link
  scheduler.py  Scheduler: the pending pool and what goes into a round
  sessions.py   SessionStore: rows and KV blocks of the live sessions
  runner.py     DraftRunner: the speculator on the GPU, one round at a time

Target side:
  client.py     DrafterClient: attach / submit / collect / release / detach
  router.py     DrafterRouter: the same over several engines, of any kind
  (vllm/v1/worker/gpu/spec_decode/remote/speculator.py adapts it to the
  model runner's speculator interface.)
"""
