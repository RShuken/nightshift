# Nightshift

**Batch LLM observability on rented GPUs.**

Your observability tool samples a few percent of traces, because judging them with a
frontier API costs real money per call. Nightshift bursts a pool of consumer GPUs on the
[Dispersed](https://dispersed.com) network instead, judges **100%** of your traces in one
pass, and tears the pool down. You pay for minutes of GPU time, not per token.

```
┌─ INGEST ────────┐   ┌─ ACCUMULATE ──┐   ┌─ SCHEDULE ─┐
│ proxy captures  │──▶│ daily JSONL   │──▶│ nightly    │
│ every LLM call  │   │               │   │ trigger    │
└─────────────────┘   └───────────────┘   └─────┬──────┘
                                                │
┌─ REPORT ────────┐   ┌─ CLUSTER ─────┐   ┌─ ENGINE ───▼──────────────┐
│ scorecard +     │◀──│ labels →      │◀──│ cook N GPUs on Dispersed  │
│ failure modes   │   │ ranked modes  │   │ judge every episode       │
│ + real $ spent  │   │ (1 GPU, secs) │   │ tear the fleet down       │
└─────────────────┘   └───────────────┘   └───────────────────────────┘
```

## The five pieces

| Piece | What it is | Where |
|---|---|---|
| **Ingest** | Transparent OpenAI/Anthropic proxy. One env var, zero code changes. | `nightshift/proxy.py` |
| **Accumulate** | Append-only daily JSONL. | `nightshift/store.py` |
| **Schedule** | Nightly trigger. | `nightshift/scheduler.py` |
| **Engine** | Provision → burst-judge → tear down. | `nightshift/{fleet,judge}.py` |
| **Report** | Scorecard, failure taxonomy, real cost. | `nightshift/{cluster,report,cost}.py` |
| *Control* | MCP surface — ask Claude to run and query it. | `nightshift/mcp_server.py` |

## Quick start

```bash
cp .env.example .env        # add your Dispersed pk_/sk_ pair
./ns status                 # account, corpus, live GPUs

./ns proxy                  # terminal 1: ingest
export OPENAI_BASE_URL=http://localhost:8788/v1    # terminal 2: your app

./ns run --fleet 4          # burst now
./ns report                 # last report
./ns schedule --at 02:00    # nightly
./ns kill                   # panic button: cancel every GPU
```

## As an MCP server

```bash
claude mcp add nightshift -- python3 -m nightshift.mcp_server
```

Then ask: *"what did my agent screw up yesterday?"* → `failure_modes` →
`episodes_in_cluster` for the receipts.

## The unit of judgment

Not a session (a 450KB transcript blows any context window) and not a message (too small
to have an outcome), but a **task episode**: one user request through to its resolution.
That's the thing that can independently succeed or fail.

Each episode gets one schema-constrained judgment:

```json
{"scores": {"instruction_following": 1-5, "correctness": 1-5, "efficiency": 1-5},
 "verdict": "pass|fail", "failure_label": "short reusable phrase", "evidence": "..."}
```

Pass 2 then reads *only the labels* — so clustering is one cheap call regardless of
corpus size.

## Operational notes

Nodes are strangers' hardware and **they do drop**. Two consequences shaped the design:

- The burst is a **work queue**, not static sharding — a dead node's episodes get picked
  up by whoever is free.
- Teardown is **belt-and-braces**: `atexit` + signal handlers + `./ns kill` reconcile that
  cancels orphans from crashed runs. PERSISTENT jobs bill until cancelled, so leaking a
  fleet overnight is the expensive mistake.

The judge image is ~22.7 GB, so cold start is dominated by the node's image pull. Cook
more nodes than you need and race them; `wait_ready` proceeds with whoever arrives.

## Status

Built at a hackathon. Traces are sent to third-party GPU nodes **unredacted** — a
deliberate MVP tradeoff, not a recommendation for production.
