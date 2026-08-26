"""Provision, track and tear down a fleet of llama.cpp judges on Dispersed.

Teardown is belt-and-braces: atexit + signal handlers + a reconcile() that
cancels ANY nightshift job still running, even from a crashed prior process.
PERSISTENT jobs bill until cancelled, so leaking them is the expensive failure.
"""
import atexit
import base64
import json
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

from . import config
from .dispersed import request

_LIVE = set()          # job uuids this process created
_LOCK = threading.Lock()


def _log(msg):
    print(f"[fleet] {msg}", flush=True)


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------
def cook(n=None, title_suffix=""):
    """Cook n llama.cpp jobs on 5090s. Returns list of job uuids."""
    n = n or config.FLEET_SIZE
    uuids = []
    for i in range(n):
        body = {
            "title": f"{config.JOB_TITLE_PREFIX}-{title_suffix or 'run'}-{i}",
            "task": "PERSISTENT",
            "max_timeout_run_ms": None,
            "gpu_registry_uuid": config.GPU_5090_UUID,
            "max_timeout_start_ms": config.MAX_TIMEOUT_START_MS,
            "max_timeout_assign_ms": config.MAX_TIMEOUT_ASSIGN_MS,
            "max_retry_count": 3,
        }
        status, payload = request(
            "POST", f"/v1/job-recipes/{config.LLAMACPP_RECIPE_UUID}/cook", body=body
        )
        if status not in (200, 201):
            _log(f"cook {i} FAILED {status}: {json.dumps(payload)[:400]}")
            continue
        uuid = payload.get("uuid") or payload.get("data", {}).get("uuid")
        if uuid:
            uuids.append(uuid)
            with _LOCK:
                _LIVE.add(uuid)
            _log(f"cooked {i+1}/{n} -> {uuid}")
    return uuids


def job_runs(job_uuid):
    status, payload = request("GET", "/v1/job-runs", query={"limit": "50"})
    if status != 200:
        return []
    return [r for r in payload.get("data", []) if r.get("job_uuid") == job_uuid]


def all_runs(limit=50):
    status, payload = request("GET", "/v1/job-runs", query={"limit": str(limit)})
    return payload.get("data", []) if status == 200 else []


def endpoint_of(run):
    """Turn a job_run into a base URL + auth header, or None if not ready."""
    urls = run.get("node_urls") or []
    if not urls:
        return None
    # prefer the container port we care about
    pick = None
    for u in urls:
        if str(u.get("protocol", "")).lower() in ("http", "https"):
            pick = u
            break
    pick = pick or urls[0]
    scheme = "https" if pick.get("tls") else "http"
    base = f"{scheme}://{pick['hostname']}:{int(pick['port'])}"

    headers = {"Content-Type": "application/json"}
    toks = run.get("container_access_tokens") or []
    if toks:
        tok = toks[0]
        tok = tok.get("token") if isinstance(tok, dict) else tok
        if tok:
            raw = base64.b64encode(f"duser:{tok}".encode()).decode()
            headers["Authorization"] = f"Basic {raw}"
    return {"base": base, "headers": headers, "run_uuid": run.get("uuid")}


def wait_ready(job_uuids, timeout_s=1200, poll_s=10, need=None):
    """Poll until jobs are RUNNING and their llama.cpp server answers.

    Returns (ready_endpoints, timings). Does not require all of them —
    decentralized nodes drop, so we proceed with whoever shows up.
    """
    need = need or len(job_uuids)
    t0 = time.time()
    ready, timings, seen_status = {}, {}, {}
    wanted = set(job_uuids)

    while time.time() - t0 < timeout_s:
        runs = all_runs(limit=50)
        for run in runs:
            juuid = run.get("job_uuid") or (run.get("job_snapshot") or {}).get("uuid")
            if juuid not in wanted or juuid in ready:
                continue
            st = run.get("status")
            if seen_status.get(juuid) != st:
                seen_status[juuid] = st
                _log(f"{juuid[:8]} -> {st}  (+{time.time()-t0:.0f}s)")
            if st != "RUNNING":
                continue
            ep = endpoint_of(run)
            if not ep:
                continue
            if health(ep):
                ready[juuid] = ep
                timings[juuid] = time.time() - t0
                _log(f"READY {juuid[:8]} in {timings[juuid]:.0f}s  {ep['base']}")
        if len(ready) >= need:
            break
        if ready and time.time() - t0 > 300:
            break  # some are up and we've waited long enough; go with what we have
        time.sleep(poll_s)

    return list(ready.values()), timings


def health(ep, timeout=8):
    for path in ("/health", "/v1/models"):
        try:
            req = urllib.request.Request(ep["base"] + path, headers=ep["headers"])
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status < 500:
                    return True
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return True
        except Exception:
            pass
    return False


# --------------------------------------------------------------------------
# teardown
# --------------------------------------------------------------------------
def stop(job_uuid):
    status, _ = request("PUT", f"/v1/jobs/{job_uuid}/cancel",
                        body={"reason": "nightshift batch complete"})
    return status in (200, 202, 204)


def teardown(uuids=None):
    with _LOCK:
        targets = list(uuids if uuids is not None else _LIVE)
    if not targets:
        return
    _log(f"tearing down {len(targets)} job(s)")
    for u in targets:
        ok = stop(u)
        _log(f"  cancel {u[:8]} {'ok' if ok else 'FAILED'}")
        with _LOCK:
            _LIVE.discard(u)


def reconcile():
    """Cancel every nightshift job still alive, including orphans. Safety net."""
    status, payload = request("GET", "/v1/jobs", query={"limit": "50"})
    if status != 200:
        _log(f"reconcile: cannot list jobs ({status})")
        return []
    killed = []
    for job in payload.get("data", []):
        title = job.get("title") or ""
        st = (job.get("status") or "").upper()
        if title.startswith(config.JOB_TITLE_PREFIX) and st not in (
            "COMPLETED", "CANCELLED", "FAILED"
        ):
            if stop(job["uuid"]):
                killed.append(job["uuid"])
                _log(f"reconciled orphan {job['uuid'][:8]} ({title})")
    return killed


def _panic(signum, frame):
    _log(f"signal {signum} — emergency teardown")
    teardown()
    sys.exit(130)


atexit.register(teardown)
for _s in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_s, _panic)
    except Exception:
        pass
