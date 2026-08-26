# Local judge endpoint (no remote GPUs required)

Validated end-to-end on this machine (Apple Silicon, darwin 25.5.0) on 2026-08-26.
`nightshift/judge.py` runs against it **unmodified**.

## TL;DR

```bash
llama-server \
  -m /Users/shuken/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff \
  --host 127.0.0.1 --port 8081 -c 32768 -ngl 99 --alias llama3.2-judge
```

**base_url:** `http://127.0.0.1:8081`  (judge.py appends `/v1/chat/completions`)

```python
endpoints = [{"base": "http://127.0.0.1:8081",
              "headers": {"Content-Type": "application/json"}}]
results, stats = judge.run(episodes, endpoints)
```

Wait for readiness with `curl -s http://127.0.0.1:8081/health` -> `{"status":"ok"}`
(~10-20s to load). `/v1/models` also responds, so `fleet.py`'s health probes both pass.

## Why llama.cpp and not Ollama

Ollama **is** installed and has a suitable model, but it cannot run this pipeline
unmodified. `judge.py::_call` builds its request body with only
`messages` / `max_tokens` / `temperature` — **there is no `model` field**, because the
Dispersed fleet runs a single-model llama.cpp server that defaults it.

Ollama rejects that:

```
$ curl -X POST http://localhost:11434/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"say hi"}],"max_tokens":50,"temperature":0.0}'
HTTP 400
{"error":{"message":"model is required","type":"api_error","param":null,"code":null}}
```

This is fatal, not degradable: `_call` only advances to the next `response_format`
on a 4xx, so all four variants 400 with the same `model is required` and the last
one re-raises. Every episode would exhaust `max_attempts` and land in `failures`.

llama.cpp accepts the body as-is, so it needs no shim and matches production behaviour.

## Setup that was performed

- `llama.cpp` installed via `brew install llama.cpp` (bottled, tens of MB; deps `ggml`, `openssl@3`).
- **No model was downloaded.** It is pointed at the GGUF blob Ollama already had on
  disk: `llama3.2:latest`, 3.2B params, Q4_K_M, 2.0 GB.
- Ollama's own daemon (`ollama serve`) was started during probing and is independent;
  it is not needed for the judge.

Models already local (nothing pulled):

| model | size | note |
|---|---|---|
| `llama3.2:latest` | 2.0 GB | 3.2B Q4_K_M — **used here** |
| `deepseek-r1:latest` | 4.7 GB | reasoning model, emits `<think>`; poor fit for constrained JSON |
| `dolphin-mixtral:latest` | 26 GB | 46.7B MoE, too heavy for a laptop smoke test |

## `response_format` support matrix

`judge.py` tries four variants strongest-first and caches the first that works
(`_FMT_IDX`). Tested with the real `SCHEMA` and `SYSTEM` prompt from `judge.py`.

### llama.cpp (`llama-server`, port 8081) — the one we use

| # | variant | HTTP | strict `json.loads` | correct nested shape |
|---|---|---|---|---|
| 0 | `{"type":"json_schema","json_schema":{...}}` | 200 | yes | **yes** |
| 1 | `{"type":"json_object","schema":{...}}` | 200 | yes | **yes** |
| 2 | `{"type":"json_object"}` | 200 | no (```-fenced) | no — flat keys |
| 3 | none (prompt only) | 200 | no (```-fenced) | no — flat keys |

Variant 0 succeeds on the first call, so `_FMT_IDX` stays `0` and every subsequent
request uses true schema-constrained decoding. The `[judge] response_format -> ...`
line never prints, which is the expected sign that the strongest variant held.

**Nothing 400s here.** Variants 2 and 3 return 200 but ignore the schema: they emit
flat `instruction_following` / `correctness` keys instead of the required nested
`scores` object, and wrap the JSON in a ``` fence. `judge.py`'s brace-scan fallback
(`text.find("{")` / `rfind("}")`) does recover the fence, but the resulting record is
missing `scores`, so downstream consumers would break. This only matters if you ever
force a lower variant — do not.

### Ollama (`http://localhost:11434/v1`) — for reference

With a `model` field added, all four variants return 200, but:

- variant 0 (`json_schema`) — honoured correctly, right nested shape;
- variant 1 (`json_object` + sibling `schema`) — **the `schema` key is silently ignored**,
  producing the wrong flat shape. llama.cpp honours it; Ollama does not.
- variants 2/3 — flat shape, as above.

So Ollama also never 400s on `response_format`; its only blocker is the missing `model`.

## Verified end-to-end run

`judge.run()` over 3 synthetic episodes against the local endpoint:

```
[judge] 1/3  15s  0.07 ep/s  errs=0
[judge] 2/3  33s  0.06 ep/s  errs=0
[judge] 3/3  50s  0.06 ep/s  errs=0

{"episodes": 3, "judged": 3, "failed": 0, "errors": 0, "elapsed_s": 50.1,
 "nodes": 1, "throughput_eps": 0.06,
 "prompt_tokens": 750, "completion_tokens": 239, "failures": []}
```

Sample record — correct nested schema, sensible judgment:

```json
{
  "id": "ep1",
  "request": "Rename `foo` to `bar` in utils.py. Touch nothing else.",
  "scores": {"instruction_following": 1, "correctness": 2, "efficiency": 1},
  "verdict": "fail",
  "failure_label": "strayed from task",
  "evidence": "Also reformatted main.py and removed an import in app.py",
  "_node": "http://127.0.0.1:8081"
}
```

The clean episode scored 5/5/5 `pass`, the flailing one `fail` /
`"gave up on test"` — so verdict logic, `failure_label` and `evidence` all behave.

## Caveats

- **Throughput is ~0.06 ep/s** (~17s/episode) — roughly 1000x slower than the GPU
  fleet. This is for correctness validation, not for real batches.
- `-ngl 99` puts all layers on the Metal GPU; the model needs ~2 GB of unified memory.
- Port 8081 was chosen because **5000** (`config.CONTAINER_PORT`) is occupied by macOS
  ControlCenter and **8080** by Docker on this machine.
- A 3.2B model is a weak judge. Scores are directionally right but should not be
  treated as a quality baseline for the 70B-class model the fleet runs.
- Single endpoint means `judge.run` starts one worker thread, so the work-queue
  retry/fan-out path is only lightly exercised. Pass the same base twice to smoke-test
  concurrency.
