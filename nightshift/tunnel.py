"""SSH port-forwards to llama.cpp servers.

The judge image runs llama.cpp bound to 127.0.0.1:8080 *inside* the container,
so it is not reachable over the network directly. Dispersed exposes SSH on the
node; we forward a local port through it and talk to the model over the tunnel.
"""
import atexit
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

KEY = Path.home() / ".ssh" / "nightshift"
KNOWN_HOSTS = Path.home() / ".ssh" / "nightshift_known_hosts"
REMOTE_HOST, REMOTE_PORT = "127.0.0.1", 8080
_PROCS = []


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def ensure_key():
    KNOWN_HOSTS.touch(mode=0o600, exist_ok=True)
    if not KEY.exists():
        KEY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", str(KEY), "-N", "",
                        "-q", "-C", "nightshift"], check=True)
    return (KEY.with_suffix(".pub")).read_text().strip()


def open_tunnel(hostname, port, user="duser", local_port=None, timeout=45):
    """Forward local_port -> container's 127.0.0.1:8080 over SSH. Returns base URL."""
    local_port = local_port or _free_port()
    cmd = [
        "ssh", "-i", str(KEY), "-p", str(port), f"{user}@{hostname}",
        "-N", "-T",
        "-L", f"{local_port}:{REMOTE_HOST}:{REMOTE_PORT}",
        # accept-new + a persistent per-fleet known_hosts: first sight of a node
        # is trusted, but a CHANGED key afterwards fails loudly instead of being
        # silently accepted the way StrictHostKeyChecking=no would.
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=15",
        "-o", "ConnectTimeout=20",
        "-o", "LogLevel=ERROR",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _PROCS.append(proc)

    base = f"http://127.0.0.1:{local_port}"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            err = proc.stderr.read().decode()[:300] if proc.stderr else ""
            raise RuntimeError(f"ssh tunnel died: {err}")
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=2):
                pass
            return base
        except OSError:
            time.sleep(1)
    proc.kill()
    raise TimeoutError(f"tunnel to {hostname}:{port} never opened")


def wait_model(base, headers=None, timeout=600):
    """Wait for llama.cpp to finish loading the model behind an open tunnel."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        for path in ("/health", "/v1/models"):
            try:
                req = urllib.request.Request(base + path, headers=headers or {})
                with urllib.request.urlopen(req, timeout=8) as r:
                    if r.status == 200:
                        return True
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    return True          # up, just wants auth
            except Exception:
                pass
        time.sleep(5)
    return False


def close_all():
    for p in _PROCS:
        try:
            p.kill()
        except Exception:
            pass
    _PROCS.clear()


atexit.register(close_all)
