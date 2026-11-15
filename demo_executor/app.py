from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from demo_executor.engine import DemoEngine  # noqa: E402

PORT = int(os.getenv("OSH_LOCAL_PORT", "8765"))
SESSION = secrets.token_urlsafe(32)
HTML = Path(__file__).resolve().parent / "static" / "index.html"
engine = DemoEngine()
app = FastAPI(title="OSH Local Binance Demo", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def local_only(request: Request, call_next):
    peer = request.client.host if request.client else ""
    host = request.headers.get("host", "")
    pages_origin = "https://sherzoddd2002.github.io"
    origins = {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}", pages_origin}
    if peer not in {"127.0.0.1", "::1"} or host not in {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}:
        return JSONResponse({"detail": "Localhost access only."}, status_code=403)
    origin = request.headers.get("origin")
    if origin and origin not in origins:
        return JSONResponse({"detail": "Remote browser origins blocked."}, status_code=403)
    return await call_next(request)


# Only the owned GitHub Pages origin may call this loopback service cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sherzoddd2002.github.io"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-OSH-Session"],
    max_age=600,
)


def auth(request: Request):
    if request.headers.get("x-osh-session") != SESSION:
        raise HTTPException(status_code=403, detail="Local session token invalid.")


@app.get("/", response_class=HTMLResponse)
def home():
    if not HTML.exists():
        raise HTTPException(status_code=500, detail="Local terminal UI missing.")
    return HTMLResponse(HTML.read_text(encoding="utf-8").replace("__OSH_SESSION__", SESSION))


@app.get("/api/session")
def session(request: Request):
    if request.headers.get("origin") != "https://sherzoddd2002.github.io":
        raise HTTPException(status_code=403, detail="GitHub Pages origin required.")
    return {"session": SESSION}


@app.get("/api/status")
def status(request: Request):
    auth(request)
    return engine.status()


@app.post("/api/credentials")
async def save_credentials(request: Request):
    auth(request)
    try:
        payload = await request.json()
        return engine.save_credentials(payload.get("api_key"), payload.get("api_secret"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/reset")
def reset_credentials(request: Request):
    auth(request)
    try:
        return engine.reset_credentials()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/connect")
def connect(request: Request):
    auth(request)
    try:
        return engine.connect_report()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/start")
async def start(request: Request):
    auth(request)
    try:
        payload = await request.json()
        return engine.start(confirm_demo_orders=payload.get("confirm_demo_orders") is True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/stop")
def stop(request: Request):
    auth(request)
    return engine.stop_entries()


def main():
    import uvicorn
    print(f"OSH Futures Demo Terminal: http://127.0.0.1:{PORT}")
    print("API host is hard-coded to https://demo-fapi.binance.com")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
