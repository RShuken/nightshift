"""Cook N judges and race them; nodes that fail to prepare get replaced."""
import json, sys, time
from nightshift import fleet, config
N = int(sys.argv[1]) if len(sys.argv)>1 else 3
uuids = fleet.cook(n=N, title_suffix="race")
print(f"[race] cooked {len(uuids)}", flush=True)
eps, timings = fleet.wait_ready(uuids, timeout_s=1800, poll_s=10, need=1)
fleet._LIVE.clear()
if not eps:
    print("[race] none ready", flush=True); sys.exit(1)
print(f"[race] {len(eps)} READY, cold starts: {[f'{v:.0f}s' for v in timings.values()]}", flush=True)
json.dump({"endpoints":eps,"timings":timings,"jobs":uuids},
          open(config.DATA/"fleet.json","w"), indent=1)
print("[race] saved data/fleet.json", flush=True)
