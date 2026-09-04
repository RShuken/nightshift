"""Rebuild the report from saved results — recovers a run whose post-processing died."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import json, sys
from pathlib import Path
from nightshift import cluster, config, cost, fleet, report

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path((config.RUNS/"latest").read_text().strip())
results = json.loads((d/"results.json").read_text())
stats = json.loads((d/"stats.json").read_text())
print(f"recovered {len(results)} judgments from {d.name}")

eps, jobs = fleet.load_pool()
tax = {"clusters": []}
for ep in eps:
    try:
        tax = cluster.build(results, ep); print(f"{len(tax['clusters'])} failure modes"); break
    except Exception as e: print("cluster failed on", ep["base"], type(e).__name__)

try: c = cost.for_jobs(set(jobs), stats)
except Exception: c = {"usd":0.0,"gpu_hours":0.0,"per_episode":0.0,"hourly_rate_usd":0.69,
                       "frontier_equiv_usd":None,"savings_multiple":None,"source":"unavailable"}
s = report.summarize(results, stats, tax, c)
(d/"summary.json").write_text(json.dumps(s, indent=1, default=str))
(d/"report.md").write_text(report.markdown(s))
(d/"report.html").write_text(report.html_report(s, results))
(config.RUNS/"latest").write_text(str(d))
print(report.markdown(s))
