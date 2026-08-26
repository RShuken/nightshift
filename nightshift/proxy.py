"""Ingest: a transparent OpenAI/Anthropic-compatible logging proxy.

Point your app at it and every call is captured:
    export OPENAI_BASE_URL=http://localhost:8788/v1
    export ANTHROPIC_BASE_URL=http://localhost:8788

Zero code changes in the app being observed. That's the whole trick.
"""
import json
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import config, store

app = FastAPI(title="nightshift-ingest")
_client = httpx.AsyncClient(timeout=600.0)

HOP = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}


@app.get("/nightshift/status")
async def status():
    return {"ok": True, "upstream": config.UPSTREAM_BASE, "captured": store.stats()}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def relay(path: str, request: Request):
    raw = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    url = f"{config.UPSTREAM_BASE.rstrip('/')}/{path}"
    t0 = time.time()

    try:
        req_json = json.loads(raw) if raw else None
    except Exception:
        req_json = None

    streaming = bool(isinstance(req_json, dict) and req_json.get("stream"))

    try:
        upstream = await _client.request(
            request.method, url, content=raw, headers=headers,
            params=dict(request.query_params),
        )
    except Exception as e:
        store.append({"kind": "llm_call", "path": path, "error": f"{type(e).__name__}: {e}",
                      "request": req_json, "latency_s": time.time() - t0})
        return JSONResponse({"error": {"message": str(e), "type": "proxy_error"}}, status_code=502)

    body = upstream.content
    latency = time.time() - t0

    resp_json = None
    if not streaming:
        try:
            resp_json = json.loads(body)
        except Exception:
            resp_json = None

    # capture — this is the accumulation step
    if req_json is not None or resp_json is not None:
        store.append({
            "kind": "llm_call",
            "path": path,
            "status": upstream.status_code,
            "model": (req_json or {}).get("model"),
            "latency_s": round(latency, 3),
            "streamed": streaming,
            "request": req_json,
            "response": resp_json if not streaming else {"_streamed": True},
            "usage": (resp_json or {}).get("usage"),
        })

    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP}
    return StreamingResponse(iter([body]), status_code=upstream.status_code,
                             headers=out_headers,
                             media_type=upstream.headers.get("content-type"))
