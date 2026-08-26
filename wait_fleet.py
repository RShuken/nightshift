import time, json, sys
from nightshift import fleet, config
t0=time.time()
while time.time()-t0 < 1500:
    runs=fleet.all_runs(30)
    act=[r for r in runs if r.get('status') in ('RUNNING','PREPARING','ASSIGNED')]
    run=[r for r in runs if r.get('status')=='RUNNING' and (r.get('node_urls') or [])]
    if run:
        eps=[]
        for r in run:
            ep=fleet.endpoint_of(r)
            if ep and fleet.health(ep): eps.append(ep)
        if eps:
            json.dump(eps, open(config.DATA/"live_endpoints.json","w"), indent=1)
            print(f"READY {len(eps)} endpoints after {time.time()-t0:.0f}s", flush=True)
            for e in eps: print("  ", e["base"], flush=True)
            sys.exit(0)
    if not act:
        print(f"ALL RESOLVED, none ready after {time.time()-t0:.0f}s", flush=True); sys.exit(1)
    time.sleep(15)
print("TIMEOUT", flush=True); sys.exit(2)
