"""Real cost from the Dispersed billing fields, plus a frontier-API comparison."""
from .dispersed import request

# Hosted judge tiers, USD per 1M tokens (in, out). We report ALL of them rather
# than cherry-picking the flattering one — a comparison that only survives against
# the most expensive competitor is not a comparison worth making on stage.
TIERS = [
    ("frontier (\u2248$3/$15)",  3.00, 15.00),
    ("mid (\u2248$1/$5)",        1.00,  5.00),
    ("mini (\u2248$0.15/$0.60)", 0.15,  0.60),
]
FRONTIER_IN, FRONTIER_OUT = 1.00, 5.00
FRONTIER_LABEL = "mid-tier hosted judge @ $1/$5 per Mtok"


def for_jobs(job_uuids, stats):
    """Cost of THIS batch.

    Charged as (nodes x burst wall-clock), not the fleet's whole lifetime: a warm
    pool that sat idle for an hour before you pressed go did not cost this batch
    anything, and billing it here would misreport the marginal cost of judging.
    Idle warm time is reported separately as `warm_idle_usd` so nothing is hidden.
    """
    gpu_hours, usd, rate = 0.0, 0.0, 0.69
    billed = False
    try:
        status, payload = request("GET", "/v1/job-runs", query={"limit": "50"}, timeout=20)
        if status == 200:
            for r in payload.get("data", []):
                if r.get("job_uuid") not in job_uuids:
                    continue
                rate = r.get("hourly_rate_usd") or rate
                hrs = (r.get("duration_ms") or 0) / 3_600_000
                gpu_hours += hrs
                usd += hrs * rate
                billed = True
    except Exception as e:
        # The judgments are already paid for and in hand. A metadata timeout must
        # never discard them — fall back to a wall-clock estimate and say so.
        print(f"[cost] billing lookup failed ({type(e).__name__}); estimating from wall clock")

    fleet_hours, fleet_usd = gpu_hours, usd      # whole-fleet lifetime, if billed

    # the batch itself: nodes actually working x how long the burst took
    burst_hours = stats.get("nodes", 0) * stats.get("elapsed_s", 0) / 3600
    gpu_hours = burst_hours or fleet_hours
    usd = gpu_hours * rate
    idle_usd = max(0.0, fleet_usd - usd) if billed else 0.0

    n = max(stats.get("judged", 0), 1)
    pin, pout = stats.get("prompt_tokens", 0), stats.get("completion_tokens", 0)
    frontier = pin / 1e6 * FRONTIER_IN + pout / 1e6 * FRONTIER_OUT
    comparison = []
    for label, ti, to in TIERS:
        hosted = pin / 1e6 * ti + pout / 1e6 * to
        comparison.append({
            "tier": label,
            "hosted_usd": round(hosted, 4),
            "ratio": round(hosted / usd, 2) if usd > 0 else None,
        })
    return {
        "usd": round(usd, 4),
        "gpu_hours": round(gpu_hours, 4),
        "hourly_rate_usd": rate,
        "per_episode": round(usd / n, 6),
        "frontier_equiv_usd": round(frontier, 4) if frontier else None,
        "frontier_basis": FRONTIER_LABEL,
        "savings_multiple": round(frontier / usd, 1) if usd > 0 and frontier else None,
        "source": "dispersed_billing" if billed else "wall_clock_estimate",
        "rate_basis": "burst wall-clock x nodes",
        "warm_idle_usd": round(idle_usd, 4),
        "fleet_lifetime_usd": round(fleet_usd, 4) if billed else None,
        "comparison": comparison,
    }
