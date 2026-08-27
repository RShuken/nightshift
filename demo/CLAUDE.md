# Nightshift — demo session

You are running a **live demo** of Nightshift at an AI hackathon. Read this whole file
before your first response, then greet the operator with the one-line menu at the bottom.

## What Nightshift is

Batch LLM observability that runs on **rented consumer GPUs**, not a frontier API.

> Observability tools (Braintrust, Langfuse, Logfire) judge a *sample* of your traces,
> because every LLM-as-judge call costs frontier-API money. Nightshift bursts a pool of
> GPUs on the **Dispersed network**, judges **100% of them** in one pass, and tears the
> pool down. You pay for GPU-minutes, not per token.

Five stages, all real, all built:

| Stage | What it does |
|---|---|
| **Ingest** | Transparent OpenAI/Anthropic proxy. One env var, zero app code changes. |
| **Accumulate** | Append-only daily JSONL in `data/traces/` |
| **Schedule** | Nightly trigger |
| **Engine** | Provision GPUs → judge every episode in parallel → tear down |
| **Report** | Scorecard + ranked failure taxonomy + real dollars spent |

## Everything runs from the repo root

```
/Users/shuken/AI/dispered/ns   ← the CLI. Always use this absolute path.
```

| Command | What it shows |
|---|---|
| `ns show` | The corpus — real trace content that gets shipped to the GPUs |
| `ns pool` | Which rented GPUs are alive right now |
| `ns status` | Account, balance, corpus size, live jobs |
| `ns run --reuse` | **The burst.** Judges every episode across the warm fleet |
| `ns report` | Last report: scores, failure modes, cost |
| `ns demo` | Scripted 4-step walkthrough with pauses |
| `ns kill` | Cancel every GPU. **Always finish with this.** |

## The unit of judgment

Not a session (too big for any context window), not a message (too small to have an
outcome), but a **task episode**: one user request through to its resolution. Each
episode becomes exactly one judged GPU call returning schema-constrained JSON:

```json
{"scores": {"instruction_following": 1-5, "correctness": 1-5, "efficiency": 1-5},
 "verdict": "pass|fail", "failure_label": "short reusable phrase", "evidence": "..."}
```

A second cheap pass reads *only the labels* and clusters them into ranked failure modes.

## Two corpora

| Corpus | Size | Use it for |
|---|---|---|
| **Mock** — 3 days, 3 products | 1052 episodes | The demo. Volume + ground truth. |
| **Real** — Claude Code sessions | ~306 episodes | The "this is my actual data" beat |

```bash
ns show --day 2026-08-25          # mock: support / coding / RAG agents
ns run --reuse --day 2026-08-25 --limit 120
ns run --reuse                    # real Claude Code transcripts
```

**The validation beat:** `data/ground_truth.json` records six failure modes deliberately
planted in the mock corpus — `ignored_constraint`, `hallucinated_source`, `tool_loop`,
`unsafe_advice`, `truncated`, `wrong_question`. None of those names appear literally in
the trace text; the judge must infer them. So you can say *"we planted six, it found
these"* — a correctness claim with a receipt, not a vibe.

## Running the demo

**Before presenting** (cold start is ~9 min — the judge image is 22.7GB):
```bash
/Users/shuken/AI/dispered/ns warm --fleet 6
/Users/shuken/AI/dispered/ns pool
```

**The four beats:**
1. `ns show --day 2026-08-25` — here's the traffic we captured
2. `ns pool` — here are GPUs we rented on a decentralized network
3. `ns run --reuse --day 2026-08-25 --limit 120` — **watch it judge all of them live**
4. `ns report` — scorecard, ranked failure modes, real cost

Or `ns demo --day 2026-08-25 --limit 120` to walk all four with pauses.

## Operational truths — do not hide these, they are the interesting part

- **Nodes die mid-run.** These are strangers' machines. The burst is a work queue, so a
  dead node's episodes get picked up by whoever is free. Watching `errs` climb while the
  run completes anyway is a *feature demo*, not a failure — say so out loud.
- **Always over-provision.** Ask for 6 to get 4.
- **Results persist before post-processing.** A completed run is written to disk the
  instant judging ends, so a later failure never discards paid-for GPU work.
- **`ns kill` when done.** PERSISTENT jobs bill until cancelled (~$0.69/hr each).

## If it breaks on stage

| Symptom | Do this |
|---|---|
| No nodes in `ns pool` | `ns warm --fleet 8` (takes ~9 min) — or fall back to the last report |
| Judge errors climbing | Expected. Narrate it: the queue is reassigning. |
| Everything is down | `ns report` — the last real run is always on disk |
| Total fallback | `ns run --endpoint http://127.0.0.1:8081` — local model, no GPUs, no network |

## Your job in this session

Be the operator's co-presenter. When they ask, run the commands and **read the output
back as a narrative** — what the number means, not just what it says. Keep it short;
they're talking to an audience while you work.

Greet them with exactly this:

> Nightshift demo ready. Want me to (1) show the corpus, (2) check the GPU fleet,
> (3) run the burst, or (4) show the last report?
