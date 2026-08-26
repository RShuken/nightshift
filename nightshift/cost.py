"""Real cost from the Dispersed billing fields, plus a frontier-API comparison."""
from .dispersed import request

# Conservative "cheap frontier judge" baseline (USD per 1M tokens).
FRONTIER_IN, FRONTIER_OUT = 1.00, 5.00
FRONTIER_LABEL = "cheap frontier judge @ $1/$5 per Mtok"


def for_jobs(job_uuids, stats):
    gpu_hours, usd, rate = 0.0, 0.0, 0.69
    status, payload = request("GET", "/v1/job-runs", query={"limit": "50"})
    if status == 200:
        for r in payload.get("data", []):
            if r.get("job_uuid") not in job_uuids:
                continue
            rate = r.get("hourly_rate_usd") or rate
            hrs = (r.get("duration_ms") or 0) / 3_600_000
            gpu_hours += hrs
            usd += hrs * rate

    # fall back to wall-clock if billing hasn't settled yet
    if gpu_hours == 0 and stats.get("elapsed_s"):
        gpu_hours = stats["nodes"] * stats["elapsed_s"] / 3600
        usd = gpu_hours * rate

    n = max(stats.get("judged", 0), 1)
    frontier = (stats.get("prompt_tokens", 0) / 1e6 * FRONTIER_IN +
                stats.get("completion_tokens", 0) / 1e6 * FRONTIER_OUT)
    return {
        "usd": round(usd, 4),
        "gpu_hours": round(gpu_hours, 4),
        "hourly_rate_usd": rate,
        "per_episode": round(usd / n, 6),
        "frontier_equiv_usd": round(frontier, 4) if frontier else None,
        "frontier_basis": FRONTIER_LABEL,
        "savings_multiple": round(frontier / usd, 1) if usd > 0 and frontier else None,
    }
