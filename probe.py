"""Cook 1 GPU, measure cold start + throughput, leave it up for development."""
import json, time, urllib.request, sys
from nightshift import fleet, config

t0 = time.time()
uuids = fleet.cook(n=1, title_suffix="probe")
if not uuids:
    sys.exit("cook failed")
print(f"[probe] cooked in {time.time()-t0:.1f}s: {uuids}", flush=True)

eps, timings = fleet.wait_ready(uuids, timeout_s=1500, poll_s=8)
if not eps:
    print("[probe] NEVER BECAME READY", flush=True)
    fleet.teardown(uuids)
    sys.exit(1)

ep = eps[0]
cold = list(timings.values())[0]
print(f"[probe] COLD START = {cold:.0f}s  endpoint={ep['base']}", flush=True)

# throughput test
body = {"messages":[{"role":"user","content":"Count from 1 to 40, comma separated. Nothing else."}],
        "max_tokens":220,"temperature":0}
t1=time.time()
req = urllib.request.Request(ep["base"]+"/v1/chat/completions",
        data=json.dumps(body).encode(), headers=ep["headers"], method="POST")
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    dt = time.time()-t1
    usage = out.get("usage",{})
    ct = usage.get("completion_tokens",0)
    print(f"[probe] gen {ct} tok in {dt:.1f}s = {ct/max(dt,.01):.1f} tok/s", flush=True)
    print("[probe] sample:", json.dumps(out["choices"][0]["message"]["content"])[:200], flush=True)
    print("[probe] usage:", usage, flush=True)
except Exception as e:
    print(f"[probe] completion FAILED: {type(e).__name__} {e}", flush=True)
    try: print("[probe] body:", e.read().decode()[:600], flush=True)
    except Exception: pass

json.dump({"base":ep["base"],"headers":ep["headers"],"job_uuid":uuids[0],
           "run_uuid":ep["run_uuid"],"cold_start_s":cold},
          open(config.DATA/"probe_endpoint.json","w"), indent=1)
print(f"[probe] endpoint saved. KEEPING ALIVE for dev. job={uuids[0]}", flush=True)
fleet._LIVE.clear()   # don't auto-teardown; we want it for development
