"""Render the batch result: markdown for the terminal, HTML for the artifact."""
import collections
import html
import json
import statistics
from datetime import datetime, timezone

from . import config

DIMS = ("instruction_following", "correctness", "efficiency")


def summarize(results, stats, taxonomy, cost):
    n = len(results)
    fails = [r for r in results if r.get("verdict") == "fail"]
    by_dim = {}
    for d in DIMS:
        vals = [r["scores"][d] for r in results
                if isinstance(r.get("scores"), dict) and isinstance(r["scores"].get(d), int)]
        by_dim[d] = {
            "mean": round(statistics.mean(vals), 2) if vals else None,
            "low": sum(1 for v in vals if v <= 2),
            "n": len(vals),
        }
    by_project = collections.Counter(r.get("project", "?") for r in fails)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episodes": n,
        "failures": len(fails),
        "fail_rate": round(len(fails) / n, 3) if n else 0,
        "dimensions": by_dim,
        "worst_projects": by_project.most_common(8),
        "taxonomy": taxonomy,
        "run": stats,
        "cost": cost,
    }


def markdown(s):
    L = []
    A = L.append
    A(f"# Nightshift batch report\n")
    A(f"_{s['generated_at']}_\n")
    A(f"## Headline\n")
    A(f"- **{s['episodes']}** episodes judged")
    A(f"- **{s['failures']}** failures ({s['fail_rate']*100:.1f}%)")
    r = s["run"]
    A(f"- **{r['elapsed_s']}s** wall clock across **{r['nodes']}** GPUs "
      f"({r['throughput_eps']} ep/s)")
    c = s["cost"]
    A(f"- **${c['usd']:.2f}** total  (${c['per_episode']:.5f}/episode, "
      f"{c['gpu_hours']:.2f} GPU-hours)")
    if c.get("comparison"):
        A("")
        A("## Same corpus on a hosted judge")
        A("")
        A("| judge tier | would cost | vs Nightshift |")
        A("|---|---|---|")
        for row in c["comparison"]:
            ratio = row.get("ratio")
            if ratio and ratio >= 1:
                v = f"**{ratio:.1f}x cheaper**"
            elif ratio:
                v = f"{1/ratio:.1f}x _more expensive_"
            else:
                v = "-"
            A(f"| {row['tier']} | ${row['hosted_usd']:.2f} | {v} |")
        A("")
        A("_GPU-hours are fixed: more nodes buys speed, not cost._")
    A(f"\n## Scores\n")
    A("| dimension | mean | scoring ≤2 |")
    A("|---|---|---|")
    for d, v in s["dimensions"].items():
        A(f"| {d} | {v['mean']} | {v['low']}/{v['n']} |")
    tax = s["taxonomy"].get("clusters", [])
    if tax:
        A(f"\n## Failure modes\n")
        for i, cl in enumerate(tax, 1):
            A(f"**{i}. {cl['name']}** — {cl['count']} episodes  \n{cl.get('description','')}\n")
    if s["worst_projects"]:
        A(f"\n## Worst projects\n")
        for p, k in s["worst_projects"]:
            A(f"- `{p}` — {k} failures")
    if r.get("failures"):
        A(f"\n## Unjudged ({r['failed']})\n")
        for f in r["failures"][:8]:
            A(f"- `{f['id']}`: {f['error'][:120]}")
    return "\n".join(L)


def _bar(pct, color):
    return (f'<div class="bar"><span style="width:{pct:.1f}%;background:{color}"></span></div>')


def html_report(s, results):
    tax = s["taxonomy"].get("clusters", [])
    c, r = s["cost"], s["run"]
    esc = html.escape

    cards = "".join(f"""
      <div class="card"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>"""
      for k, v in [
          ("episodes judged", f"{s['episodes']:,}"),
          ("failures", f"{s['failures']:,} ({s['fail_rate']*100:.1f}%)"),
          ("wall clock", f"{r['elapsed_s']:.0f}s"),
          ("GPUs burst", str(r["nodes"])),
          ("total cost", f"${c['usd']:.2f}"),
          ("per episode", f"${c['per_episode']:.5f}"),
      ])

    dims = "".join(f"""
      <tr><td>{esc(d)}</td><td class="num">{v['mean']}</td>
      <td>{_bar((v['mean'] or 0)/5*100, 'var(--ok)')}</td>
      <td class="num">{v['low']}/{v['n']}</td></tr>"""
      for d, v in s["dimensions"].items())

    maxc = max((cl["count"] for cl in tax), default=1)
    modes = "".join(f"""
      <div class="mode">
        <div class="mhead"><span class="rank">{i}</span>
          <span class="mname">{esc(cl['name'])}</span>
          <span class="mcount">{cl['count']}</span></div>
        {_bar(cl['count']/maxc*100, 'var(--bad)')}
        <p>{esc(cl.get('description',''))}</p>
      </div>""" for i, cl in enumerate(tax, 1))

    projects = "".join(f"<li><code>{esc(p)}</code><span>{k}</span></li>"
                       for p, k in s["worst_projects"])

    savings = ""
    if c.get("comparison"):
        rows = ""
        for row in c["comparison"]:
            ratio = row.get("ratio")
            if ratio and ratio >= 1:
                v = f"{ratio:.1f}\u00d7 cheaper"
            elif ratio:
                v = f"{1/ratio:.1f}\u00d7 more"
            else:
                v = "\u2014"
            rows += (f"<tr><td>{esc(row['tier'])}</td>"
                     f"<td class='num'>${row['hosted_usd']:.2f}</td>"
                     f"<td class='num'>{v}</td></tr>")
        savings = (f"<div class=\"savings\"><strong>${c['usd']:.2f}</strong> spent on rented GPUs. "
                   f"The same corpus judged by a hosted model:"
                   f"<table style=\"margin-top:.7rem\"><tr><th>judge tier</th>"
                   f"<th class=\"num\">cost</th><th class=\"num\">vs us</th></tr>{rows}</table>"
                   f"<p style=\"margin:.7rem 0 0;font-size:.85rem\">GPU-hours are fixed \u2014 "
                   f"more nodes buys speed, not cost.</p></div>")

    return f"""<title>Nightshift Batch Report</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b68;--line:#e4e4e1;--card:#fff;
--ok:#3d7f5d;--bad:#c1503b;--acc:#2b5c8a;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151514;--fg:#eceae4;
--mut:#9a9a94;--line:#2c2c29;--card:#1e1e1c;--ok:#63b389;--bad:#e0765e;--acc:#7fb0dd;}}}}
:root[data-theme="dark"]{{--bg:#151514;--fg:#eceae4;--mut:#9a9a94;--line:#2c2c29;
--card:#1e1e1c;--ok:#63b389;--bad:#e0765e;--acc:#7fb0dd;}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);margin:0;padding:3rem 1.5rem;
font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;}}
.wrap{{max-width:900px;margin:0 auto}}
h1{{font-size:2rem;margin:0 0 .25rem;letter-spacing:-.02em}}
.sub{{color:var(--mut);margin:0 0 2.5rem;font-size:.9rem}}
h2{{font-size:1.05rem;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);
margin:2.75rem 0 1rem;font-weight:600}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.75rem}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem}}
.card .k{{color:var(--mut);font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}}
.card .v{{font-size:1.5rem;font-weight:650;margin-top:.35rem;letter-spacing:-.02em}}
.savings{{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);
border-radius:8px;padding:1rem 1.25rem;margin-top:1rem}}
.savings em{{color:var(--acc);font-style:normal;font-weight:650}}
table{{width:100%;border-collapse:collapse}}
td,th{{padding:.6rem .5rem;border-bottom:1px solid var(--line);text-align:left}}
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.bar{{background:var(--line);border-radius:99px;height:7px;width:100%;overflow:hidden}}
.bar span{{display:block;height:100%;border-radius:99px}}
.mode{{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:1rem 1.15rem;margin-bottom:.7rem}}
.mhead{{display:flex;align-items:center;gap:.6rem;margin-bottom:.6rem}}
.rank{{background:var(--bad);color:#fff;width:22px;height:22px;border-radius:6px;
display:grid;place-items:center;font-size:.75rem;font-weight:700;flex:none}}
.mname{{font-weight:650;flex:1}}
.mcount{{color:var(--mut);font-variant-numeric:tabular-nums}}
.mode p{{margin:.6rem 0 0;color:var(--mut);font-size:.9rem}}
ul{{list-style:none;padding:0}}
ul li{{display:flex;justify-content:space-between;padding:.5rem 0;
border-bottom:1px solid var(--line)}}
code{{font-size:.85rem}}
footer{{margin-top:3rem;padding-top:1.25rem;border-top:1px solid var(--line);
color:var(--mut);font-size:.82rem}}
</style>
<div class="wrap">
<h1>Nightshift batch report</h1>
<p class="sub">{esc(s['generated_at'])} · {r['nodes']} × RTX 5090 on the Dispersed network</p>
<div class="cards">{cards}</div>
{savings}
<h2>Scores</h2>
<table><tr><th>dimension</th><th class="num">mean</th><th></th><th class="num">≤2</th></tr>
{dims}</table>
<h2>Failure modes</h2>
{modes or '<p class="sub">No failures found.</p>'}
<h2>Worst projects</h2>
<ul>{projects or '<li>none</li>'}</ul>
<footer>{s['episodes']:,} episodes · {r['prompt_tokens']:,} prompt tokens ·
{r['completion_tokens']:,} completion tokens · {c['gpu_hours']:.2f} GPU-hours ·
judged by Qwen3-27B on rented consumer GPUs.</footer>
</div>"""
