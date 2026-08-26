"""MCP control surface: ask Claude to run and query your own observability.

    claude mcp add nightshift -- python3 -m nightshift.mcp_server
"""
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import config, episodes, store
from .batch import run_batch

mcp = FastMCP("nightshift")


def _latest_dir():
    p = config.RUNS / "latest"
    if not p.exists():
        return None
    d = Path(p.read_text().strip())
    return d if d.exists() else None


@mcp.tool()
def ingest_status() -> str:
    """How many LLM calls has the proxy captured, broken down by day?"""
    s = store.stats()
    if not s:
        return "No traces captured yet. Point your app at the proxy: OPENAI_BASE_URL=http://localhost:8788/v1"
    return json.dumps({"days": s, "total": sum(s.values())}, indent=1)


@mcp.tool()
def preview_episodes(limit_sessions: int = 20) -> str:
    """Count the judgeable task episodes available, without spending anything."""
    eps = episodes.collect(limit_sessions=limit_sessions)
    return json.dumps({
        "episodes": len(eps),
        "sources": {s: sum(1 for e in eps if e["source"] == s)
                    for s in {e["source"] for e in eps}},
        "median_chars": sorted(e["chars"] for e in eps)[len(eps)//2] if eps else 0,
    }, indent=1)


@mcp.tool()
def run_nightly_batch(fleet_size: int = 4, limit_episodes: int = 0,
                      limit_sessions: int = 0) -> str:
    """Provision GPUs on Dispersed, judge every accumulated episode, cluster the
    failures, tear the fleet down, and return the report. This spends real money."""
    out = run_batch(
        fleet_size=fleet_size,
        limit_episodes=limit_episodes or None,
        limit_sessions=limit_sessions or None,
    )
    if "error" in out:
        return f"Batch failed: {out['error']}"
    return out["markdown"]


@mcp.tool()
def last_report() -> str:
    """The most recent batch report, without re-running anything."""
    d = _latest_dir()
    if not d:
        return "No batch has been run yet."
    return (d / "report.md").read_text()


@mcp.tool()
def failure_modes() -> str:
    """The ranked failure taxonomy from the most recent batch."""
    d = _latest_dir()
    if not d:
        return "No batch has been run yet."
    s = json.loads((d / "summary.json").read_text())
    return json.dumps(s.get("taxonomy", {}).get("clusters", []), indent=1)


@mcp.tool()
def episodes_in_cluster(cluster_name: str, limit: int = 5) -> str:
    """Concrete failing episodes belonging to one failure mode, with evidence."""
    d = _latest_dir()
    if not d:
        return "No batch has been run yet."
    rows = json.loads((d / "results.json").read_text())
    hits = [r for r in rows if r.get("cluster", "").lower() == cluster_name.lower()][:limit]
    return json.dumps([{k: r.get(k) for k in
                        ("id", "project", "request", "failure_label", "evidence", "scores")}
                       for r in hits], indent=1)


if __name__ == "__main__":
    mcp.run()
