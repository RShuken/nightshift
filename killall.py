"""Cancel every nightshift job, retrying until the API confirms none are alive."""
import time
from nightshift.dispersed import request
from nightshift import config

def alive():
    s, p = request("GET", "/v1/jobs", query={"limit": "50"}, timeout=25, retries=6)
    return [j for j in p.get("data", [])
            if (j.get("title") or "").startswith(config.JOB_TITLE_PREFIX)
            and (j.get("status") or "").upper() not in ("COMPLETED", "CANCELLED", "FAILED")]

for attempt in range(8):
    try:
        jobs = alive()
    except Exception as e:
        print(f"list failed ({type(e).__name__}), retrying..."); time.sleep(5); continue
    if not jobs:
        print("\n✅ CONFIRMED: no nightshift jobs alive"); break
    print(f"\nattempt {attempt+1}: {len(jobs)} alive")
    for j in jobs:
        try:
            s, _ = request("PUT", f"/v1/jobs/{j['uuid']}/cancel",
                           body={"reason": "demo over"}, timeout=25, retries=5)
            print(f"  cancel {j['uuid'][:8]} {j.get('title','')[:28]:<28} -> {s}")
        except Exception as e:
            print(f"  cancel {j['uuid'][:8]} FAILED {type(e).__name__}")
    time.sleep(6)
else:
    print("\n⚠️  could not confirm — check https://console.dispersed.com/jobs")
