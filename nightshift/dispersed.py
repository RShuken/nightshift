"""Minimal Dispersed API client: HMAC-SHA256 request signing.

Canonical string (7 pipe-delimited parts):
    publicKey|timestamp|nonce|METHOD|pathname|queryString|bodySha256
"""
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
import urllib.request
from pathlib import Path


def _load_env():
    """Load .env from the project root so any entrypoint works without a wrapper."""
    envf = Path(__file__).resolve().parent.parent / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

BASE_URL = os.environ.get("DISPERSED_API_BASE_URL", "https://api.dispersed.com")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _canonicalize_json(value):
    if isinstance(value, dict):
        return {k: _canonicalize_json(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_json(v) for v in value]
    return value


def _canonical_query(query):
    """Sort keys, then values within a key; RFC3986 encode; join with & (no leading ?)."""
    if not query:
        return ""
    parts = []
    for key in sorted(query):
        values = query[key]
        values = [values] if isinstance(values, str) else list(values)
        for value in sorted(str(v) for v in values):
            parts.append(
                f"{urllib.parse.quote(str(key), safe='')}={urllib.parse.quote(value, safe='')}"
            )
    return "&".join(parts)


def auth_headers(method, pathname, query=None, body=None, public_key=None, secret_key=None):
    public_key = public_key or os.environ["DISPERSED_PUBLIC_KEY"]
    secret_key = secret_key or os.environ["DISPERSED_SECRET_KEY"]

    timestamp = str(int(time.time() * 1000))
    nonce = secrets.token_hex(16)

    if body is not None:
        canonical_body = json.dumps(
            _canonicalize_json(body), separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        body_sha256 = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
    else:
        body_sha256 = EMPTY_SHA256

    canonical_string = "|".join(
        [
            public_key,
            timestamp,
            nonce,
            method.upper(),
            pathname,
            _canonical_query(query),
            body_sha256,
        ]
    )
    signature = hmac.new(
        secret_key.encode("utf-8"), canonical_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return {
        "X-API-Key": public_key,
        "X-Time": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }, canonical_string


def request(method, pathname, query=None, body=None, timeout=20, retries=4):
    """Signed request with exponential backoff.

    Nonces are single-use, so each attempt re-signs. Shorter default timeout +
    more attempts beats one long hang: the control plane occasionally stalls a
    connection under load and a fresh connection succeeds immediately.
    """
    last = None
    for attempt in range(retries):
        try:
            return _request_once(method, pathname, query, body, timeout)
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 8))
    raise last


def _request_once(method, pathname, query=None, body=None, timeout=30):
    headers, canonical = auth_headers(method, pathname, query=query, body=body)

    url = BASE_URL + pathname
    if query:
        url += "?" + _canonical_query(query)

    data = None
    if body is not None:
        data = json.dumps(
            _canonicalize_json(body), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        return exc.code, payload
