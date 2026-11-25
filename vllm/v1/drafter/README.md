# Drafter engine

A drafter (draft model or draft head) served from its own process and GPU, for
any number of target engines. It is to a drafter what an inference engine is to
an LLM: clients send work whenever they have some, the engine batches whatever
is pending, and nothing is reserved for anyone.

    python -m vllm.v1.drafter.serve --method eagle3 --model <head> --target-model <target> -k 3
    python -m vllm.v1.drafter.status host:7700

A target uses it with

    speculative_config = {"num_speculative_tokens": k, "drafter_endpoint": ["a:7700", "b:7700"]}

and then loads no drafter and keeps no draft KV. It names no method and no draft
model: the engines say what they serve, and they may be of different kinds.

## One interface for every drafter

A target and an engine exchange the same things whatever the drafter is:

- the target sends the tokens of its step and their verify result, plus the
  hidden states of the layers the engine reads, if it reads any;
- the engine answers with k draft tokens per session and their acceptance
  confidences (ones if the drafter cannot estimate them).

What the target has to know about an engine is its `DrafterInfo`, which the
engine serves in its `Status`: the kind and checkpoint (for logs and checks), the
target a head was trained for, k, and the layers whose hidden states it reads
(none for a draft model). The target reads it before it loads its model, exports
the union of the layers its engines read, and gives each engine its own.

Many targets may use one engine and one target many engines. Which engine drafts
for a request is a pairing made above both: the request names it in its
`vllm_xargs` (`{"drafter": "a:7700"}`), and the target obeys; a request that
names none goes to the engine with the fewest of this target's requests. A head
only drafts for the target it was trained for (the engine refuses others); a
draft model drafts for any target with its vocabulary.

All engines of a target have the same k, the target's `num_speculative_tokens`:
the target's scheduler gives every request k draft slots before it knows who
drafts for it. How many of the k are drafted is decided above, like the pairing:

- per request, `vllm_xargs` `{"draft_tokens": 2}`, for the request's life;
- per engine, at any time, `SetDraftTokens` (`client.set_draft_tokens()` or
  `python -m vllm.v1.drafter.status host:7700 --draft-tokens 2`).

The lower of the two holds. A round runs as many draft steps as its most
demanding session needs, so drafting time falls with the largest number in the
round (DFlash drafts in one pass, for it only the padding changes). What a
session did not get is padding with confidence 0. This is meant for targets with
adaptive verification: it weighs the confidences, so it does not verify padding
(except in a request's first steps or right after its number fell, as its budget
goes by the confidences of the step before). Without it padding is verified and
rejected, which costs tokens, not correctness.

## Vocabulary

| term | meaning |
|---|---|
| client | an attached target engine (`links.Client` in the engine, `client.DrafterClient` in the target; a target has one per engine) |
| session | one sequence the engine keeps drafter KV for; named by the client, a row in the engine |
| step | one target step of a client: tokens, verify result (and hidden states) of its sessions; one `DraftRequest`, one reply |
| pending pool | the steps the engine has taken and not yet drafted (`Scheduler`) |
| round | one `propose()` of the drafter over the steps the scheduler picked; what a step is to an inference engine |

## Modules

| file | role | inference-engine analogue |
|---|---|---|
| `protocol.py` | messages and tensor layouts on the wire | |
| `config.py` | `DrafterConfig`: every knob, same names as the `serve` flags | `EngineArgs` |
| `engine.py` | `DrafterEngine`: the loop, control messages, rounds, status | `EngineCore` |
| `links.py` | `Client` record and the NCCL link to a client | |
| `scheduler.py` | `Scheduler`: pending pool, capacity of a round, policy | `Scheduler` |
| `sessions.py` | `SessionStore`: rows and KV blocks of live sessions | `KVCacheManager` |
| `runner.py` | `DraftRunner`: the stock speculator on the GPU | `GPUModelRunner` |
| `client.py` | `DrafterClient`: attach / submit / collect / release / detach, `query_status` | |
| `router.py` | `DrafterRouter`: the same interface over several engines of any kind; keeps a session on the engine it was paired with, cuts a step by engine, merges the replies | |
| `transport.py` | the NCCL link between a client and an engine, and NCCL's settings for it | |
| `serve.py`, `status.py` | command lines | |

`vllm/v1/worker/gpu/spec_decode/remote/speculator.py` (`RemoteSpeculator`) adapts
`DrafterRouter` to the model runner: `InputBatch` in, draft tokens out. On the
target the method is "remote" (set by `drafter_endpoint`), which stands for "the
drafts come from somewhere else": k draft slots per request, no drafter rows.

## Target-side interface

    infos = drafters.describe()                        # what each engine serves; waits for them
    layers = drafters.use_hidden_layers(default)       # the layers to export, for all engines
    drafters.attach(Attach(...))
    drafters.open(session, endpoint)                   # a request and the engine it was paired with
    pending = drafters.submit(request, step, hidden)   # leaves now, nothing waits
    ...                                                # other GPU work may be enqueued here
    reply = drafters.collect(pending)                  # the stream waits for the reply here

`hidden` maps a layer to the hidden states of the step's tokens; each engine is
sent the layers it reads.
`submit` sends the payload on the current CUDA stream and receives the reply on
a stream of its own; `collect` makes the current stream wait for it. What the
caller enqueues between the two runs while the engines draft. Today
`RemoteSpeculator.propose()` calls them back to back.

## Life of a step

1. Target, `RemoteSpeculator.propose()`: describe the step (`DraftRequest` with
   its `StepLayout`) and `submit()` it. With several engines the router cuts the
   step by engine first; a step adaptive verification sized on the GPU is read
   back for that (`StepLayout.exact_starts`), a wait for the sizing kernel, not
   for the forward pass. The request leaves on ZMQ at once; the payload send is
   enqueued on the target's CUDA stream behind its forward pass, the reply
   receive on the client's reply stream. Nothing blocks.
2. Engine, `_receive`: the request goes to the client's inbox.
3. Engine, `_take` / `_begin`: the sessions get rows and KV blocks
   (`SessionStore`), the payload receive starts on the client's stream, the
   `Step` enters the pending pool.
4. Engine, `Scheduler.schedule`: when the GPU is done with the last round, the
   landed steps that fit make the next round (FCFS or priority). A step that is
   alone in the engine skips this and is launched ahead of its payload.
5. Engine, `_round`: `DraftRunner.propose()` over all the steps, then one reply
   per step on its client's link.
6. Target: `collect()` makes the target's stream wait for the reply.

`Release` (a request finished on the target) frees a session; `Detach`, or 60 s
of silence, frees a client.

## What the unit of work is, today

A step is one unit: the sessions of one target step are drafted in one round
and answered in one reply, because the target's next forward pass needs all of
them. Three places encode that, and they are the ones to change for
session-level service (a target submits sessions, each comes back when it is
done):

- `protocol.DraftRequest` / the reply: one payload and one reply per step; a
  reply would have to say which sessions it carries.
- `scheduler.Step` and `Scheduler._pool`: one entry per client; it would hold
  sessions, and a round would take any subset.
- the target: `RemoteSpeculator.propose()` submits and collects back to back
  and returns drafts for the whole batch. The interface already separates the
  two; the model runner would have to call them at different times, and the
  target's scheduler decide which requests run in which forward pass.

`DraftRunner.propose()` and `SessionStore` already work per session.

## Limits

Greedy drafts, one k (the most a session can get) for all the engines of a target, target TP 1. A
target's prefix cache is off by default with `drafter_endpoint` (a draft model
can follow hits through `DraftRequest.prefixes`, a head cannot).

A link costs GPU memory on both sides. NCCL's default of 64 collective channels
per communicator took about 1.9 GiB per link over InfiniBand; `transport.py`
sets it to one channel (about 35 MiB), in the engine and in the target, unless
the environment says otherwise. The setting is per process. If the engine dies, stops answering
status queries or drops the target, the target exits (`drafter_timeout`).

A target that dies in the middle of a step leaves transfers that never
complete. They sit on that client's own CUDA stream (payload receives, reply
sends), except a round launched ahead of a payload that then never comes,
which holds the engine's stream. After `--idle-timeout` (60 s) of silence the
engine drops the client and aborts its link, which ends those transfers; in
three tries (target killed mid-run, one client, prelaunch on) a new target
then attached and ran normally, and without the abort only one try of three
did. Until the timeout, other clients may be held up. A link is freed
(`ncclCommAbort`) only when nothing is in flight on it, or when its client is
taken for dead: aborting a link whose peer is alive but not receiving blocked
every CUDA call of the engine.
