# Local judge (no GPUs, no network)

`ns run --endpoint <url>` points the whole pipeline at any OpenAI-compatible
server instead of provisioning GPUs. Useful for validating the pipeline, and as
the last-resort fallback if the network is unavailable during a demo.

## llama.cpp (recommended)

```bash
brew install llama.cpp                       # macOS; see llama.cpp docs for Linux
llama-server -m /path/to/model.gguf --port 8081 --jinja
./ns run --endpoint http://127.0.0.1:8081 --day 2026-08-25 --limit 20
```

Any instruct GGUF works. A 3B model validates the pipeline in a couple of minutes;
it is *not* a quality baseline for the 27B fleet model.

> If you already have Ollama installed, its model blobs are plain GGUF files under
> `~/.ollama/models/blobs/` and can be passed to `llama-server -m` directly — no
> second download.

## Why not Ollama's own server?

The fleet runs llama.cpp in *router mode*, so `judge.py` sends an explicit `model`
field discovered from `/v1/models`. Ollama also requires `model`, but its
`/v1/models` returns Ollama tags rather than a single served model, and it ignores
the `schema` sibling in `response_format: json_object`. llama.cpp honors both. Use
llama.cpp for parity with what runs on the fleet.

## `response_format` support (measured against llama.cpp)

| variant | works | notes |
|---|---|---|
| `json_schema` | ✅ | true constrained decoding — this is what the fleet uses |
| `json_object` + `schema` | ✅ | same |
| `json_object` alone | ⚠️ | returns 200 but ignores the schema → flat keys |
| none | ⚠️ | prose with a code fence |

`judge.py` tries these strongest-first and remembers which one the server accepted.
