"""The nightly batch: provision → judge → cluster → report → tear down.

This is the whole product in one function. Everything else is a surface onto it.
"""
import json
import time
from datetime import datetime, timezone

from . import cluster, config, cost, episodes, fleet, judge, report


def run_batch(day=None, fleet_size=None, use_claude_code=True,
              limit_sessions=None, limit_episodes=None, reuse=None, keep_alive=False,
              endpoint=None):
    """endpoint: an explicit OpenAI-compatible base URL (local model, or a tunnel).
    When given, no GPUs are provisioned — useful for validating the pipeline."""
    t0 = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = config.RUNS / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[batch] {m}", flush=True)

    # 1. gather
    eps = episodes.collect(use_claude_code=use_claude_code, day=day,
                           limit_sessions=limit_sessions)
    if limit_episodes:
        eps = eps[:limit_episodes]
    log(f"{len(eps)} episodes to judge")
    if not eps:
        return {"error": "no episodes found"}

    # 2. provision
    jobs = []
    if endpoint:
        endpoints = [{"base": endpoint.rstrip("/"),
                      "headers": {"Content-Type": "application/json"},
                      "run_uuid": None}]
        log(f"using explicit endpoint {endpoints[0]['base']} (no GPUs provisioned)")
    elif reuse:
        endpoints, jobs = (reuse, []) if isinstance(reuse, list) else fleet.load_pool()
        log(f"reusing {len(endpoints)} warm node(s)")
        if not endpoints:
            return {"error": "no warm nodes — run './ns warm' first"}
        keep_alive = True
    else:
        n = fleet_size or config.FLEET_SIZE
        jobs = fleet.cook(n=n, title_suffix=stamp[-6:])
        quorum = max(1, int(n * 0.6))   # don't burst on a fleet of one
        endpoints, timings = fleet.wait_ready(jobs, timeout_s=1800, poll_s=10,
                                              need=quorum, grace_s=240)
        log(f"{len(endpoints)}/{n} nodes ready "
            f"({', '.join(f'{v:.0f}s' for v in timings.values())})")
        if not endpoints:
            fleet.teardown(jobs)
            return {"error": "no nodes became ready"}

    try:
        # 3. burst
        results, stats = judge.run(eps, endpoints)
        log(f"judged {stats['judged']}/{stats['episodes']} in {stats['elapsed_s']}s")
        # Persist FIRST. Everything after this is cheap post-processing, and none
        # of it is worth losing a paid-for GPU run to.
        (outdir / "results.json").write_text(json.dumps(results, indent=1, default=str))
        (outdir / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
        log(f"raw results saved -> {outdir}/results.json")

        # 4. cluster
        # Nodes die mid-run. Try every endpoint before giving up on the taxonomy.
        taxonomy = {"clusters": []}
        for ep in endpoints:
            try:
                taxonomy = cluster.build(results, ep)
                log(f"{len(taxonomy.get('clusters', []))} failure modes")
                break
            except Exception as e:
                log(f"clustering failed on {ep['base']}: {type(e).__name__}")
                taxonomy = {"clusters": [], "error": str(e)}
    finally:
        if jobs and not keep_alive:
            fleet.teardown(jobs)

    # 5. report
    try:
        c = cost.for_jobs(set(jobs), stats)
    except Exception as e:
        log(f"cost unavailable ({type(e).__name__}) — reporting without it")
        c = {"usd": 0.0, "gpu_hours": 0.0, "per_episode": 0.0, "hourly_rate_usd": 0.69,
             "frontier_equiv_usd": None, "savings_multiple": None, "source": "unavailable"}
    summary = report.summarize(results, stats, taxonomy, c)
    summary["wall_clock_total_s"] = round(time.time() - t0, 1)

    (outdir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    md = report.markdown(summary)
    (outdir / "report.md").write_text(md)
    (outdir / "report.html").write_text(report.html_report(summary, results))
    latest = config.RUNS / "latest"
    latest.write_text(str(outdir))

    log(f"report → {outdir}")
    return {"summary": summary, "outdir": str(outdir), "markdown": md}
