"""The nightly batch: provision → judge → cluster → report → tear down.

This is the whole product in one function. Everything else is a surface onto it.
"""
import json
import time
from datetime import datetime, timezone

from . import cluster, config, cost, episodes, fleet, judge, report


def run_batch(day=None, fleet_size=None, use_claude_code=True,
              limit_sessions=None, limit_episodes=None, reuse=None, keep_alive=False):
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
    if reuse:
        endpoints = reuse
        log(f"reusing {len(endpoints)} warm node(s)")
    else:
        n = fleet_size or config.FLEET_SIZE
        jobs = fleet.cook(n=n, title_suffix=stamp[-6:])
        endpoints, timings = fleet.wait_ready(jobs, timeout_s=1800, poll_s=10, need=1)
        log(f"{len(endpoints)}/{n} nodes ready "
            f"({', '.join(f'{v:.0f}s' for v in timings.values())})")
        if not endpoints:
            fleet.teardown(jobs)
            return {"error": "no nodes became ready"}

    try:
        # 3. burst
        results, stats = judge.run(eps, endpoints)
        log(f"judged {stats['judged']}/{stats['episodes']} in {stats['elapsed_s']}s")

        # 4. cluster
        try:
            taxonomy = cluster.build(results, endpoints[0])
            log(f"{len(taxonomy.get('clusters', []))} failure modes")
        except Exception as e:
            log(f"clustering failed: {type(e).__name__}: {e}")
            taxonomy = {"clusters": [], "error": str(e)}
    finally:
        if jobs and not keep_alive:
            fleet.teardown(jobs)

    # 5. report
    c = cost.for_jobs(set(jobs), stats) if jobs else cost.for_jobs(set(), stats)
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
