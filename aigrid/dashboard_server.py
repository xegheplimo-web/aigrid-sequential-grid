"""Standalone AIGrid panel server (localhost only, read-only).

Runs the AI RECOMMENDATION vs ACTUAL EXECUTION panel on :8099.
Usage: .venv-aigrid/Scripts/python.exe -m aigrid.dashboard_server --state-dir user_data/aigrid_state
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse

from .dashboard_api import init_dashboard_api, router

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>AIGrid Panel</title>
<style>
 body{font-family:Consolas,monospace;background:#0d1117;color:#c9d1d9;margin:20px}
 h1{font-size:18px} h2{font-size:14px;color:#8b949e;margin-top:24px}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin:10px 0}
 .ok{color:#3fb950}.warn{color:#d29922}.bad{color:#f85149}.dim{color:#8b949e}
 table{border-collapse:collapse;width:100%} td,th{border:1px solid #30363d;padding:4px 8px;font-size:12px}
 pre{background:#0d1117;border:1px solid #30363d;padding:8px;font-size:11px;white-space:pre-wrap}
</style></head><body>
<h1>AIGrid — AI RECOMMENDATION vs ACTUAL EXECUTION <span class="dim">(read-only, dry-run)</span></h1>
<div class="card" id="ai"><h2>AI RECOMMENDATION</h2><pre>loading…</pre></div>
<div class="card" id="exec"><h2>ACTUAL EXECUTION STATE</h2><pre>loading…</pre></div>
<div class="card" id="events"><h2>RISK / EXECUTION EVENTS</h2><pre>loading…</pre></div>
<script>
async function refresh(){
 try{
  const s = await (await fetch('/api/aigrid/status')).json();
  document.querySelector('#ai pre').textContent = JSON.stringify(s.ai_recommendation, null, 1);
  document.querySelector('#exec pre').textContent = JSON.stringify(s.actual_execution, null, 1);
 }catch(e){ document.querySelector('#ai pre').textContent = 'ERROR: '+e; }
 try{
  const ev = await (await fetch('/api/aigrid/events?limit=30')).json();
  document.querySelector('#events pre').textContent = JSON.stringify(ev.events, null, 1);
 }catch(e){}
}
refresh(); setInterval(refresh, 5000);
</script></body></html>"""


def create_app(state_dir: str) -> FastAPI:
    app = FastAPI(title="AIGrid Panel", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    init_dashboard_api(state_dir)
    app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default="user_data/aigrid_state")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    import uvicorn

    app = create_app(args.state_dir)
    Path(args.state_dir).mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
