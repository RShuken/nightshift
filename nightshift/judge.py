"""The burst: fan every episode across the GPU fleet, one constrained-JSON judgment each.

Work-queue rather than static sharding, because decentralized nodes drop mid-run
and a dead shard must be picked up by whoever is free.
"""
import json
import queue
import threading
import time
import urllib.error
import urllib.request

from . import config

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "instruction_following": {"type": "integer", "minimum": 1, "maximum": 5},
                "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
                "efficiency": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["instruction_following", "correctness", "efficiency"],
        },
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "failure_label": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["scores", "verdict", "failure_label", "evidence"],
}

SYSTEM = """You are a strict evaluator of AI agent transcripts. You judge whether the \
assistant actually accomplished what the user asked.

Score 1-5 on:
- instruction_following: did it do what was asked, not something adjacent?
- correctness: were its claims and actions actually right?
- efficiency: did it get there without flailing, redundant tool calls, or dead ends?

verdict is "fail" if ANY score <= 2, otherwise "pass".
failure_label: if fail, a SHORT reusable phrase naming the failure mode (3-6 words, \
e.g. "ignored explicit constraint", "hallucinated file path", "looped on same error"). \
If pass, use "none".
evidence: one short quote or specific observation supporting your judgment.

Respond with JSON only."""


def _prompt(ep):
    return (f"# Task the user asked for\n{ep['request']}\n\n"
            f"# What the assistant did\n{ep['transcript']}\n\n"
            f"Evaluate this episode.")


# llama.cpp has shipped three different spellings of constrained decoding.
# Try strongest first, degrade on 4xx, and always keep the prompt-level fallback.
_FORMATS = [
    {"type": "json_schema", "json_schema": {"name": "judgment", "schema": SCHEMA}},
    {"type": "json_object", "schema": SCHEMA},
    {"type": "json_object"},
    None,
]
_FMT_LOCK = threading.Lock()
_FMT_IDX = [0]


def _post(ep_url, headers, body, timeout):
    req = urllib.request.Request(ep_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _call(ep_url, headers, episode, timeout=180, model=None):
    base = {
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": _prompt(episode)}],
        "max_tokens": config.JUDGE_MAX_TOKENS,
        "temperature": config.JUDGE_TEMPERATURE,
    }
    if model:
        base["model"] = model     # required by llama.cpp router mode
    # Qwen3 is a reasoning model: left on, it spends the whole token budget on
    # hidden thinking and returns EMPTY content. Off, it is ~5x faster and ~5x
    # cheaper per judgment, and constrained decoding gives us the JSON directly.
    base["chat_template_kwargs"] = {"enable_thinking": False}
    start = _FMT_IDX[0]
    out = None
    for i in range(start, len(_FORMATS)):
        body = dict(base)
        if _FORMATS[i] is not None:
            body["response_format"] = _FORMATS[i]
        try:
            out = _post(ep_url, headers, body, timeout)
            if i != _FMT_IDX[0]:
                with _FMT_LOCK:
                    _FMT_IDX[0] = i          # remember what this server accepts
                print(f"[judge] response_format -> {_FORMATS[i]}", flush=True)
            break
        except urllib.error.HTTPError as e:
            if e.code >= 500 or i == len(_FORMATS) - 1:
                raise
            continue
    if out is None:
        raise RuntimeError("all response_format variants rejected")
    text = out["choices"][0]["message"]["content"]
    usage = out.get("usage", {})
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s < 0 or e < 0:
            raise ValueError(f"no JSON in response: {text[:200]}")
        verdict = json.loads(text[s:e + 1])
    return verdict, usage


def run(episodes, endpoints, max_attempts=3, progress_every=10):
    """Judge every episode across the fleet. Returns (results, stats)."""
    if not endpoints:
        raise RuntimeError("no endpoints")

    q = queue.Queue()
    for ep in episodes:
        q.put((ep, 0))
    results, failures = [], []
    lock = threading.Lock()
    counters = {"done": 0, "err": 0, "in_tok": 0, "out_tok": 0}
    t0 = time.time()
    total = len(episodes)

    def worker(slot):
        url, headers, model = slot["base"], slot["headers"], slot.get("model")
        while True:
            try:
                episode, attempt = q.get_nowait()
            except queue.Empty:
                return
            try:
                verdict, usage = _call(url, headers, episode, model=model)
                rec = {**{k: episode[k] for k in
                          ("id", "source", "session", "project", "request")},
                       "chars": episode.get("chars"),
                       "n_turns": episode.get("n_turns"),
                       **verdict, "_node": url}
                with lock:
                    results.append(rec)
                    counters["done"] += 1
                    counters["in_tok"] += usage.get("prompt_tokens", 0)
                    counters["out_tok"] += usage.get("completion_tokens", 0)
                    n = counters["done"]
                if n % progress_every == 0 or n == total:
                    el = time.time() - t0
                    print(f"[judge] {n}/{total}  {el:.0f}s  "
                          f"{n/max(el,.01):.2f} ep/s  errs={counters['err']}", flush=True)
            except Exception as e:
                with lock:
                    counters["err"] += 1
                if attempt + 1 < max_attempts:
                    q.put((episode, attempt + 1))   # another worker will retry it
                else:
                    with lock:
                        failures.append({"id": episode.get("id"),
                                         "error": f"{type(e).__name__}: {e}"[:300]})
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in endpoints]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - t0
    stats = {
        "episodes": total, "judged": len(results), "failed": len(failures),
        "errors": counters["err"], "elapsed_s": round(elapsed, 1),
        "nodes": len(endpoints),
        "throughput_eps": round(len(results) / max(elapsed, .01), 3),
        "prompt_tokens": counters["in_tok"], "completion_tokens": counters["out_tok"],
        "failures": failures[:20],
    }
    return results, stats
