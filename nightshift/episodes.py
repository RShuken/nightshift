"""Normalize raw traces into 'task episodes' — the unit of judgment.

An episode is one user request through to its resolution: the thing that can
independently succeed or fail. Sessions are too big to judge whole (a 450KB
transcript blows any sane context window), and single messages are too small
to have an outcome.

Two sources:
  * Claude Code transcripts (~/.claude/projects/**/*.jsonl)
  * proxy-captured llm_call records (nightshift's own ingest)
"""
import json
from pathlib import Path

from . import config, store

MAX = config.EPISODE_CHAR_BUDGET


def _text_of(content):
    """Flatten an Anthropic-style content field to text, noting tool activity."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    parts = []
    for b in content:
        if not isinstance(b, dict):
            parts.append(str(b))
            continue
        t = b.get("type")
        if t == "text":
            parts.append(b.get("text", ""))
        elif t == "thinking":
            continue
        elif t == "tool_use":
            args = json.dumps(b.get("input", {}), default=str)
            parts.append(f"[TOOL CALL {b.get('name')}] {args[:600]}")
        elif t == "tool_result":
            c = b.get("content")
            body = _text_of(c) if not isinstance(c, str) else c
            err = " ERROR" if b.get("is_error") else ""
            parts.append(f"[TOOL RESULT{err}] {body[:600]}")
        elif "text" in b:
            parts.append(str(b["text"]))
    return "\n".join(p for p in parts if p)


def _clip(s, budget=MAX):
    if len(s) <= budget:
        return s
    head = budget * 2 // 3
    return s[:head] + f"\n\n...[{len(s)-budget} chars elided]...\n\n" + s[-(budget - head):]


# --------------------------------------------------------------------------
# Claude Code transcripts
# --------------------------------------------------------------------------
def from_transcript(path):
    """Split one session JSONL into episodes."""
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    episodes, cur = [], None
    session = path.stem
    project = path.parent.name

    for r in rows:
        typ = r.get("type")
        msg = r.get("message") or {}
        role = msg.get("role") or (typ if typ in ("user", "assistant") else None)
        if role not in ("user", "assistant"):
            continue
        text = _text_of(msg.get("content"))
        if not text.strip():
            continue

        # a real user turn (not a tool_result echo) starts a new episode
        is_tool_echo = text.lstrip().startswith("[TOOL RESULT")
        if role == "user" and not is_tool_echo:
            if cur and cur["turns"]:
                episodes.append(cur)
            cur = {"source": "claude_code", "session": session, "project": project,
                   "request": text.strip()[:4000], "turns": [],
                   "ts": r.get("timestamp")}
            continue
        if cur is None:
            continue
        cur["turns"].append(f"{role.upper()}: {text}")

    if cur and cur["turns"]:
        episodes.append(cur)

    out = []
    for i, e in enumerate(episodes):
        body = "\n\n".join(e["turns"])
        if len(body) < 80:          # nothing happened; not judgeable
            continue
        e["id"] = f"{project}/{session[:8]}#{i}"
        e["transcript"] = _clip(body)
        e["n_turns"] = len(e["turns"])
        e["chars"] = len(body)
        e.pop("turns")
        out.append(e)
    return out


def from_claude_code(root=None, limit_sessions=None):
    root = Path(root or Path.home() / ".claude" / "projects")
    files = sorted(root.rglob("*.jsonl"))
    if limit_sessions:
        files = files[:limit_sessions]
    eps = []
    for f in files:
        eps.extend(from_transcript(f))
    return eps


# --------------------------------------------------------------------------
# proxy-captured calls
# --------------------------------------------------------------------------
def from_proxy(day=None):
    eps = []
    for i, rec in enumerate(store.load(day)):
        if rec.get("kind") != "llm_call":
            continue
        req, resp = rec.get("request") or {}, rec.get("response") or {}
        msgs = req.get("messages") or []
        user = next((_text_of(m.get("content")) for m in reversed(msgs)
                     if m.get("role") == "user"), "")
        answer = ""
        ch = resp.get("choices") or resp.get("content")
        if isinstance(ch, list) and ch:
            answer = _text_of(ch[0].get("message", {}).get("content") if isinstance(ch[0], dict)
                              and "message" in ch[0] else ch)
        body = f"USER: {user}\n\nASSISTANT: {answer}"
        eps.append({
            "id": f"proxy/{day or store.day_key()}#{i}",
            "source": "proxy", "session": rec.get("path", "?"),
            "project": rec.get("model") or "proxy",
            "request": user[:4000], "transcript": _clip(body),
            "n_turns": len(msgs), "chars": len(body), "ts": rec.get("ts"),
            "usage": rec.get("usage"), "latency_s": rec.get("latency_s"),
        })
    return eps


def collect(use_claude_code=True, day=None, limit_sessions=None):
    eps = from_proxy(day)
    if use_claude_code:
        eps.extend(from_claude_code(limit_sessions=limit_sessions))
    return eps
