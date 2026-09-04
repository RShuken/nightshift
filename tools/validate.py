"""Did the judge actually recover the failure modes we planted?

The mock corpus has six modes planted in it, recorded in data/ground_truth.json.
None of the mode names appear literally in the trace text, so the judge has to
infer them. This scores recall per mode and reports the confusion honestly.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from nightshift import config

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path((config.RUNS / "latest").read_text().strip())
results = json.loads((d / "results.json").read_text())
truth = json.loads((config.DATA / "ground_truth.json").read_text())

# episode ids look like proxy/<day>#<idx>; ground truth keys are <day>#<idx>
def key(r):
    i = r.get("id", "")
    return i.split("/", 1)[1] if "/" in i else i

planted = {k: v for k, v in truth.items() if v != "none"}
seen = {key(r): r for r in results}
overlap = [k for k in seen if k in truth]

print(f"judged episodes            {len(results)}")
print(f"matched to ground truth    {len(overlap)}")
if not overlap:
    sys.exit("no overlap — did this run use the mock corpus?")

tp = fp = fn = tn = 0
per_mode = defaultdict(lambda: {"planted": 0, "caught": 0})
for k in overlap:
    r, gt = seen[k], truth[k]
    failed = r.get("verdict") == "fail"
    if gt != "none":
        per_mode[gt]["planted"] += 1
        if failed:
            tp += 1; per_mode[gt]["caught"] += 1
        else:
            fn += 1
    else:
        fp += 1 if failed else 0
        tn += 1 if not failed else 0

prec = tp / (tp + fp) if tp + fp else 0
rec = tp / (tp + fn) if tp + fn else 0
print(f"\nDETECTION (did it flag planted failures as 'fail'?)")
print(f"  caught      {tp:>4}   missed {fn:>4}")
print(f"  false alarm {fp:>4}   clean  {tn:>4}")
print(f"  precision {prec:.0%}   recall {rec:.0%}")

print(f"\nRECALL BY PLANTED MODE")
for mode, s in sorted(per_mode.items(), key=lambda x: -x[1]["planted"]):
    r_ = s["caught"] / s["planted"] if s["planted"] else 0
    bar = "█" * round(r_ * 24)
    print(f"  {mode:<24} {s['caught']:>3}/{s['planted']:<3} {r_:>4.0%} {bar}")

sm = (d / "summary.json")
if sm.exists():
    clusters = json.loads(sm.read_text()).get("taxonomy", {}).get("clusters", [])
    if clusters:
        print(f"\nMODES THE SYSTEM NAMED ON ITS OWN ({len(clusters)}), vs {len(set(planted.values()))} planted")
        for i, c in enumerate(clusters, 1):
            print(f"  {i}. {c['name']:<34} {c['count']:>4} episodes")
