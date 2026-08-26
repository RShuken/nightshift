import time
from nightshift.dispersed import request
t0=time.time()
while time.time()-t0<600:
    s,p=request('GET','/v1/job-runs',query={'limit':'10'})
    for r in p.get('data',[]):
        if (r.get('job_snapshot') or {}).get('title')=='nightshift-judge-sshtest':
            urls=r.get('node_urls') or []
            print(f"{time.time()-t0:5.0f}s {r['status']:<10} urls={len(urls)} {urls}",flush=True)
            if urls or r['status'] in ('FAILED','CANCELLED','COMPLETED'): raise SystemExit(0)
            break
    time.sleep(8)
