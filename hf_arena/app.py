"""Local-only HTTP API for the HF Model Arena.

Binds 127.0.0.1:8766 only (the H96 C2 terminal keeps 8765 untouched).
Only the owned GitHub Pages origin may call it cross-origin, and every mutating
endpoint requires a per-process session token obtained from /api/session.
"""
from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .config import LOCAL_PORT, PAGES_ORIGIN
from .engine import ArenaEngine

SESSION = secrets.token_urlsafe(32)
engine = ArenaEngine()
app = FastAPI(title="OSH HF Model Arena (local)", docs_url=None,
              redoc_url=None, openapi_url=None)


@app.middleware("http")
async def local_only(request: Request, call_next):
    peer = request.client.host if request.client else ""
    host = request.headers.get("host", "")
    allowed_hosts = {f"127.0.0.1:{LOCAL_PORT}", f"localhost:{LOCAL_PORT}"}
    if peer not in {"127.0.0.1", "::1"} or host not in allowed_hosts:
        return JSONResponse({"detail": "Localhost access only."}, status_code=403)
    origin = request.headers.get("origin")
    if origin and origin != PAGES_ORIGIN:
        return JSONResponse({"detail": "Remote browser origins blocked."}, status_code=403)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[PAGES_ORIGIN],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-OSH-HF-Session"],
    max_age=600,
)


def auth(request: Request):
    if request.headers.get("x-osh-hf-session") != SESSION:
        raise HTTPException(status_code=403, detail="Local session token invalid.")


@app.get("/", response_class=HTMLResponse)
def home():
    path = Path(__file__).resolve().parent / "static" / "index.html"
    if path.exists():
        return HTMLResponse(path.read_text(encoding="utf-8").replace("__HF_SESSION__", SESSION))
    return HTMLResponse("<h3>OSH HF Arena local engine running.</h3>")


@app.get("/api/session")
def session(request: Request):
    if request.headers.get("origin") != PAGES_ORIGIN:
        raise HTTPException(status_code=403, detail="GitHub Pages origin required.")
    return {"session": SESSION, "port": LOCAL_PORT, "version": "2026-11-15-v1"}


@app.get("/api/status")
def status(request: Request):
    auth(request)
    return engine.status()


@app.post("/api/credentials")
async def credentials(request: Request):
    auth(request)
    try:
        payload = await request.json()
        return engine.save_credentials(payload.get("api_key"), payload.get("api_secret"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/reset")
def reset(request: Request):
    auth(request)
    return engine.reset_credentials()


@app.post("/api/connect")
def connect(request: Request):
    auth(request)
    try:
        return engine.connect()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/replay")
async def replay(request: Request):
    auth(request)
    try:
        payload = await request.json()
        days = int(payload.get("days", 90))
        if not 7 <= days <= 365:
            raise ValueError("days must be between 7 and 365")
        return engine.run_replay(days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/start")
async def start(request: Request):
    auth(request)
    try:
        payload = await request.json()
        return engine.start_demo(
            champion=str(payload.get("champion", "")),
            confirm_orders=payload.get("confirm_orders") is True,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/stop")
def stop(request: Request):
    auth(request)
    return engine.stop()


@app.post("/api/kill")
def kill(request: Request):
    auth(request)
    return engine.kill()


def main():
    import uvicorn
    print(f"OSH HF Model Arena: http://127.0.0.1:{LOCAL_PORT}")
    print(f"Host pinned to testnet.binancefuture.com (NON-production). No live fallback.")
    print("Order sending is OFF by default; explicit confirmation required.")
    uvicorn.run(app, host="127.0.0.1", port=LOCAL_PORT, log_level="warning")


if __name__ == "__main__":
    main()
