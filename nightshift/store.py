"""Accumulation: append-only daily JSONL. This is the 'batches a day of requests' layer."""
import json
import threading
import time
from datetime import datetime, timezone

from . import config

_W = threading.Lock()


def day_key(ts=None):
    dt = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def path_for(day=None):
    return config.TRACES / f"{day or day_key()}.jsonl"


def append(record):
    record.setdefault("ts", time.time())
    p = path_for(day_key(record["ts"]))
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _W:
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return p


def days():
    return sorted(p.stem for p in config.TRACES.glob("*.jsonl"))


def load(day=None):
    p = path_for(day)
    if not p.exists():
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def stats():
    return {d: sum(1 for _ in open(path_for(d), encoding="utf-8")) for d in days()}
