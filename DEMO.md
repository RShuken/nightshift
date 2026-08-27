# Nightshift — demo runbook

## Before you present (do this 20+ minutes early)

```bash
./ns warm --fleet 5        # cold start is ~9 min: image is 22.7GB
./ns pool                  # confirm nodes answered
```

Leave it running. Idle cost is ~$0.69/hr per node. `./ns kill` when you're done.

> Cook more nodes than you need. Nodes are strangers' hardware and some fail to
> pull the image; `wait_ready` proceeds with whoever arrives.

## The 3-minute demo

**1. The problem (20s, no screen).**
"Your observability tool samples traces, because judging them costs frontier-API
money per call. So you're flying on 2% of your data."

**2. Ingest is real (30s).**
```bash
./ns proxy                                        # terminal 1
export OPENAI_BASE_URL=http://localhost:8788/v1   # terminal 2
```
Make one call from any app. Then:
```bash
curl -s localhost:8788/nightshift/status
```
→ the call was captured. One env var, zero code changes.

**3. The burst (60s) — the money shot.**
```bash
./ns run --reuse
```
Live progress: `[judge] 150/1052  118s  1.27 ep/s  errs=9` across N GPUs at once.
Say the number out loud: *every* trace, not a sample.

**4. The report (60s).**
```bash
./ns report                   # terminal
open data/runs/latest/report.html
```
Scorecard, then the failure taxonomy: "your agent fails in these N ways, ranked."

**5. The receipt (20s).**
Point at the cost line. Real dollars from Dispersed's own billing fields, next to
what the same corpus would have cost on a frontier judge.

## Ask Claude instead (strong alternative close)

```bash
claude mcp add nightshift -- python3 -m nightshift.mcp_server
```
Then, in Claude Code:
- *"what did my agent screw up yesterday?"* → `failure_modes`
- *"show me examples of the worst one"* → `episodes_in_cluster`

Demoing an observability tool by **asking a question in English** lands harder
than a dashboard tour.

## The validation beat (use if judges are technical)

The mock corpus has six failure modes planted in it, listed in
`data/ground_truth.json`. None of the mode names appear literally in the text —
the judge has to infer them. So you can say: *"we planted six, it found these"* —
that's a correctness claim, not a vibe.

## Corpora

| corpus | size | use |
|---|---|---|
| mock (3 days, 3 products) | 1052 episodes | the demo — volume + ground truth |
| real Claude Code sessions | 306 episodes | the "this is my actual data" beat |

```bash
./ns run --reuse --day 2026-08-25      # one mock day
./ns run --reuse --limit 200           # cap it to keep the demo short
```

## If something breaks

| symptom | fix |
|---|---|
| no nodes ready | `./ns warm --fleet 8` — some nodes always fail the pull |
| `warm pool: 0 reconnected` | jobs were cancelled; re-warm |
| judge errors climbing | one node died; the work queue reassigns automatically |
| everything is on fire | `./ns run --endpoint http://127.0.0.1:8081` (local model, no GPUs) |

**Always finish with `./ns kill`.** PERSISTENT jobs bill until cancelled.
