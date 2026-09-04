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
def cook(n=None, title_suffix="", n_parallel=None, ctx=None):
    """Provision n llama.cpp judges via POST /v1/jobs.

    IMPORTANT: this uses direct job creation, NOT /v1/job-recipes/{uuid}/cook.
    The cook endpoint reaches RUNNING with an EMPTY node_urls[], which makes the
    container unreachable. Direct create populates node_urls with the SSH proxy
    host/port. That difference is the whole ballgame.
    """
    from .tunnel import ensure_key
    n = n or config.FLEET_SIZE
    pub = ensure_key()
    env = dict(config.LLAMA_ENV)
    env["LLAMA_ARG_N_PARALLEL"] = str(n_parallel or config.LLAMA_N_PARALLEL)
    env["LLAMA_ARG_CTX_SIZE"] = str(ctx or config.LLAMA_CTX)

    uuids = []
    for i in range(n):
        body = {
            "task": "PERSISTENT",
            "title": f"{config.JOB_TITLE_PREFIX}-{title_suffix or 'run'}-{i}",
            "max_timeout_run_ms": None,
            "cpu_count": 0,
            "gpu_count": 1,
            "min_vram_gb": 23,
            "min_ram_gb": 0,
            "min_storage_gb": 0,
            "gpu_registry_uuid": config.GPU_5090_UUID,
            "max_timeout_start_ms": config.MAX_TIMEOUT_START_MS,
            "max_timeout_assign_ms": config.MAX_TIMEOUT_ASSIGN_MS,
            "max_retry_count": 3,
            "parameters": {"type": "docker", "parameters": {
                "image": config.LLAMA_IMAGE,
                "tag": "latest",
                "env": env,
                # port 22 MUST be here: publishing a ports list REPLACES the
                # default set, and if SSH is not published the node never reports
                # node_urls at all — the container becomes permanently unreachable.
                "ports": [22, config.CONTAINER_PORT],
                "sshkey": pub,
                "allowed_ips": ["0.0.0.0/0"],
            }},
        }
        status, payload = request("POST", "/v1/jobs", body=body)
        if status not in (200, 201):
            _log(f"create {i} FAILED {status}: {json.dumps(payload)[:300]}")
            continue
        uuid = payload.get("uuid")
        if uuid:
            uuids.append(uuid)
            with _LOCK:
                _LIVE.add(uuid)
            _log(f"created {i+1}/{n} -> {uuid}")
    return uuids



def all_runs(limit=50):
    status, payload = request("GET", "/v1/job-runs", query={"limit": str(limit)})
    return payload.get("data", []) if status == 200 else []


def endpoint_of(run):
    """Open an SSH tunnel to the run's llama.cpp server and return its local base URL.

    node_urls advertises SSH (the model binds 127.0.0.1:8080 inside the container,
    so it is not directly reachable). We forward a local port through that SSH hop.
    """
    from . import tunnel
    urls = run.get("node_urls") or []

    # Preferred: Dispersed proxies the published container port straight to a
    # public host:port. No tunnel, no child process, nothing to die mid-demo.
    http = next((u for u in urls
                 if str(u.get("description")) == str(config.CONTAINER_PORT)), None)
    if http:
        scheme = "https" if http.get("tls") else "http"
        ep = {"base": f"{scheme}://{http['hostname']}:{int(http['port'])}",
              "headers": {"Content-Type": "application/json"},
              "run_uuid": run.get("uuid"), "direct": True}
        toks = run.get("container_access_tokens") or []
        if toks:
            tok = toks[0].get("token") if isinstance(toks[0], dict) else toks[0]
            if tok:
                import base64 as _b64
                ep["headers"]["Authorization"] = "Basic " + _b64.b64encode(
                    f"duser:{tok}".encode()).decode()
        if health(ep):
            discover_model(ep)
            return ep
        _log(f"direct http {ep['base']} not answering — falling back to ssh tunnel")

    ssh = next((u for u in urls if u.get("description") == "ssh"), None)
    if not ssh:
        return None
    try:
        base = tunnel.open_tunnel(ssh["hostname"], int(ssh["port"]))
    except Exception as e:
        _log(f"tunnel to {ssh['hostname']}:{ssh['port']} failed: {type(e).__name__}: {e}")
        return None
    ep = {"base": base, "headers": {"Content-Type": "application/json"},
          "run_uuid": run.get("uuid"), "ssh": f"{ssh['hostname']}:{ssh['port']}"}
    discover_model(ep)
    return ep


def wait_ready(job_uuids, timeout_s=1200, poll_s=10, need=None, grace_s=240):
    """Poll until jobs are RUNNING and their llama.cpp server answers.

    Nodes prepare at wildly different speeds (22GB image pull on strangers'
    hardware), so: wait for `need` nodes, but once the FIRST one lands, give
    stragglers `grace_s` to join rather than bursting on a fleet of one.
    """
    need = len(job_uuids) if need is None else need
    t0 = time.time()
    ready, timings, seen_status = {}, {}, {}
    first_ready_at = None
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
        if first_ready_at and time.time() - first_ready_at > grace_s:
            _log(f"grace window closed with {len(ready)}/{need} ready — proceeding")
            break
        if ready and first_ready_at is None:
            first_ready_at = time.time()
        # everything already resolved (failed/cancelled)? stop waiting.
        live = {r.get("job_uuid") for r in runs
                if r.get("status") in ("ASSIGNED", "PREPARING", "RUNNING")}
        if not (wanted & live) and not ready:
            _log("all jobs resolved without becoming ready")
            break
        time.sleep(poll_s)

    return list(ready.values()), timings


def discover_model(ep, timeout=20):
    """llama.cpp router mode requires an explicit model name on every request."""
    try:
        req = urllib.request.Request(ep["base"] + "/v1/models", headers=ep["headers"])
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read()).get("data") or []
        if data:
            ep["model"] = data[0].get("id")
            return ep["model"]
    except Exception as e:
        _log(f"model discovery failed on {ep.get('ssh')}: {type(e).__name__}")
    return None


def health(ep, timeout=10):
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


# --------------------------------------------------------------------------
# warm-pool persistence (demo support: pre-warm before, burst on stage)
# --------------------------------------------------------------------------
def save_pool(endpoints, jobs):
    """Persist SSH coordinates, NOT local tunnel URLs.

    Tunnels are child processes of whoever opened them; their 127.0.0.1 ports
    die with that process. Only the ssh host:port survives across runs.
    """
    import json as _json
    (config.DATA / "fleet.json").write_text(_json.dumps({
        "jobs": jobs,
        "nodes": [{"ssh": e.get("ssh"), "direct": e.get("base") if e.get("direct") else None,
                   "model": e.get("model"), "run_uuid": e.get("run_uuid")}
                  for e in endpoints if e.get("ssh") or e.get("direct")],
    }, indent=1))


def load_pool(verify=True):
    """Re-open tunnels to a previously warmed fleet. Returns (endpoints, jobs)."""
    import json as _json
    from . import tunnel
    f = config.DATA / "fleet.json"
    if not f.exists():
        return [], []
    d = _json.loads(f.read_text())
    jobs = d.get("jobs", [])
    eps = []
    for n in d.get("nodes", []):
        if n.get("direct"):
            ep = {"base": n["direct"], "headers": {"Content-Type": "application/json"},
                  "run_uuid": n.get("run_uuid"), "direct": True, "model": n.get("model")}
            if not verify or health(ep):
                discover_model(ep)
                eps.append(ep)
                continue
            _log(f"warm pool: direct {n['direct']} dead, trying ssh")
        if not n.get("ssh"):
            continue
        host, _, port = n["ssh"].rpartition(":")
        try:
            base = tunnel.open_tunnel(host, int(port))
        except Exception as e:
            _log(f"warm pool: tunnel to {n['ssh']} failed ({type(e).__name__})")
            continue
        ep = {"base": base, "headers": {"Content-Type": "application/json"},
              "run_uuid": n.get("run_uuid"), "ssh": n["ssh"]}
        if not verify or health(ep):
            discover_model(ep)
            eps.append(ep)
        else:
            _log(f"warm pool: {n['ssh']} tunnel open but model not answering")
    _log(f"warm pool: {len(eps)} node(s) reconnected")
    return eps, jobs
