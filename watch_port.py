import time
from nightshift.dispersed import request
RUN=None
t0=time.time()
while time.time()-t0<900:
    s,p=request('GET','/v1/job-runs',query={'limit':'10'})
    for r in p.get('data',[]):
        snap=r.get('job_snapshot') or {}
        if snap.get('title')=='nightshift-judge-porttest':
            urls=r.get('node_urls') or []
            print(f"{time.time()-t0:5.0f}s {r['status']:<10} urls={len(urls)} {urls}", flush=True)
            if urls or r['status'] in ('FAILED','CANCELLED'):
                raise SystemExit(0)
            break
    time.sleep(10)
print("timeout", flush=True)
