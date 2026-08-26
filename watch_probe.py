import json, time, urllib.request, sys
from nightshift import fleet, config
JOB="01a04047-3a57-4aac-aa54-3083e126f509"
fleet._LIVE.clear()
eps,timings = fleet.wait_ready([JOB], timeout_s=1800, poll_s=10)
fleet._LIVE.clear()
if not eps: sys.exit("[probe] never ready")
ep=eps[0]; cold=list(timings.values())[0]
print(f"[probe] COLD START {cold:.0f}s -> {ep['base']}", flush=True)
body={"messages":[{"role":"user","content":"Count 1 to 40 comma separated."}],"max_tokens":220,"temperature":0}
t=time.time()
try:
    r=urllib.request.Request(ep["base"]+"/v1/chat/completions",data=json.dumps(body).encode(),headers=ep["headers"],method="POST")
    with urllib.request.urlopen(r,timeout=180) as resp: out=json.loads(resp.read())
    dt=time.time()-t; u=out.get("usage",{})
    print(f"[probe] {u.get('completion_tokens')} tok in {dt:.1f}s = {u.get('completion_tokens',0)/max(dt,.01):.1f} tok/s",flush=True)
    print("[probe] usage",u,flush=True)
    print("[probe] text",repr(out["choices"][0]["message"]["content"])[:180],flush=True)
except Exception as e:
    print("[probe] gen FAILED",type(e).__name__,e,flush=True)
    try: print(e.read().decode()[:500],flush=True)
    except Exception: pass
json.dump({"base":ep["base"],"headers":ep["headers"],"job_uuid":JOB,"cold_start_s":cold},
          open(config.DATA/"probe_endpoint.json","w"),indent=1)
print("[probe] saved, keeping alive",flush=True)
fleet._LIVE.clear()
