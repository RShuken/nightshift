"""Pass 2: turn ~N raw failure labels into a ranked taxonomy of failure modes.

Cheap by construction — it reads only the short labels, never the transcripts,
so it's one call on one GPU regardless of corpus size.
"""
import collections
import json
import urllib.request

from . import config

SYSTEM = """You group AI-agent failure labels into a small taxonomy of distinct failure modes.

Given a list of short failure labels, produce 4-8 clusters. Each cluster gets:
- name: a crisp 2-5 word name for the failure mode
- description: one sentence on what goes wrong and why it matters
- members: the exact input labels belonging to this cluster

Every input label must appear in exactly one cluster. Merge synonyms aggressively —
"ignored constraint" and "did not follow instruction" are the same mode.

Respond with JSON only: {"clusters":[{"name":...,"description":...,"members":[...]}]}"""


def build(results, endpoint, timeout=240):
    labels = [r.get("failure_label", "").strip() for r in results
              if r.get("verdict") == "fail" and r.get("failure_label", "").strip().lower()
              not in ("", "none")]
    if not labels:
        return {"clusters": [], "n_failures": 0, "note": "no failures found"}

    counts = collections.Counter(labels)
    uniq = [l for l, _ in counts.most_common(200)]

    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Failure labels (with frequency):\n" +
             "\n".join(f"- {l}  (x{counts[l]})" for l in uniq)},
        ],
        "max_tokens": 2000, "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    if endpoint.get("model"):
        body["model"] = endpoint["model"]
    body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(endpoint["base"] + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=endpoint["headers"], method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    text = out["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        parsed = json.loads(text[s:e + 1])

    clusters = parsed.get("clusters", [])
    label_to_cluster = {}
    for c in clusters:
        c["count"] = sum(counts.get(m, 0) for m in c.get("members", []))
        for m in c.get("members", []):
            label_to_cluster[m] = c["name"]
    clusters.sort(key=lambda c: -c["count"])

    # attach cluster back onto each failing result
    for r in results:
        if r.get("verdict") == "fail":
            r["cluster"] = label_to_cluster.get(r.get("failure_label", "").strip(), "unclustered")

    return {"clusters": clusters, "n_failures": len(labels),
            "n_unique_labels": len(counts)}
