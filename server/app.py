"""
FastAPI server exposing the OpenEnv-compliant contract review environment.
Endpoints: /health, /reset, /step, /state, /ws (WebSocket), /docs, /web
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, ValidationError
from typing import Optional

from models.action import ContractAction
from server.environment import ContractReviewEnvironment

app = FastAPI(
    title="Contract Review Environment",
    description=(
        "An OpenEnv-compliant RL environment for AI agents to learn "
        "contract review, risk identification, and legal document analysis."
    ),
    version="1.0.0",
)

# ─── Shared single-session environment (HTTP endpoints) ───────────────────────
# For WebSocket, each connection gets its own instance.
_http_env = ContractReviewEnvironment()


# ─── Request / Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task1"
    contract_id: Optional[str] = None


class StepRequest(BaseModel):
    action: ContractAction


# ─── HTTP Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return RedirectResponse("/web")


@app.get("/health")
def health():
    return {"status": "healthy", "environment": "contract-review-env", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    try:
        obs = _http_env.reset(
            task_id=req.task_id or "task1",
            contract_id=req.contract_id,
        )
        return {"observation": obs.model_dump(), "done": False, "reward": 0.0, "info": {}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    try:
        obs, reward = _http_env.step(req.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.value,
            "done": obs.done,
            "info": {
                "reason": reward.reason,
                "is_terminal": reward.is_terminal,
                "final_score": reward.final_score,
                "cumulative_reward": obs.cumulative_reward,
            },
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    try:
        s = _http_env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── WebSocket Endpoint (primary — used by OpenEnv Python client) ──────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket session — each connection gets its own env instance.
    Message protocol (JSON):
      Client → Server: {"type": "reset", "task_id": "task1"} 
                     | {"type": "step", "action": {...}}
                     | {"type": "state"}
      Server → Client: {"type": "reset"|"step"|"state"|"error", "data": {...}}
    """
    await websocket.accept()
    env = ContractReviewEnvironment()  # isolated per connection

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "data": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                try:
                    obs = env.reset(
                        task_id=msg.get("task_id", "task1"),
                        contract_id=msg.get("contract_id"),
                    )
                    await websocket.send_json({
                        "type": "reset",
                        "data": {
                            "observation": obs.model_dump(),
                            "done": False,
                            "reward": 0.0,
                            "info": {},
                        },
                    })
                except ValueError as e:
                    await websocket.send_json({"type": "error", "data": str(e)})

            elif msg_type == "step":
                try:
                    action_data = msg.get("action", {})
                    action = ContractAction(**action_data)
                    obs, reward = env.step(action)
                    await websocket.send_json({
                        "type": "step",
                        "data": {
                            "observation": obs.model_dump(),
                            "reward": reward.value,
                            "done": obs.done,
                            "info": {
                                "reason": reward.reason,
                                "is_terminal": reward.is_terminal,
                                "final_score": reward.final_score,
                                "cumulative_reward": obs.cumulative_reward,
                            },
                        },
                    })
                except (ValidationError, RuntimeError) as e:
                    await websocket.send_json({"type": "error", "data": str(e)})

            elif msg_type == "state":
                try:
                    s = env.state()
                    await websocket.send_json({"type": "state", "data": s.model_dump()})
                except RuntimeError as e:
                    await websocket.send_json({"type": "error", "data": str(e)})

            else:
                await websocket.send_json({
                    "type": "error",
                    "data": f"Unknown message type '{msg_type}'. Use: reset, step, state",
                })

    except WebSocketDisconnect:
        pass  # clean disconnect


# ─── Web UI ───────────────────────────────────────────────────────────────────

@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Contract Review Environment</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;background:#f9f9f9}
  h1{font-size:1.5rem;font-weight:600;margin-bottom:.25rem}
  p.sub{color:#666;margin-bottom:1.5rem}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
  .card{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:1rem}
  label{font-size:.85rem;font-weight:500;color:#555;display:block;margin-bottom:.25rem}
  select,input,textarea{width:100%;padding:.4rem .5rem;border:1px solid #ccc;border-radius:4px;font-size:.9rem;margin-bottom:.75rem;box-sizing:border-box}
  button{padding:.45rem 1rem;border:none;border-radius:4px;cursor:pointer;font-size:.9rem}
  .btn-primary{background:#1a73e8;color:#fff}
  .btn-primary:hover{background:#1557b0}
  pre{background:#f4f4f4;border-radius:4px;padding:.75rem;font-size:.78rem;overflow:auto;max-height:400px;white-space:pre-wrap;margin:0}
  .status{display:inline-block;padding:.2rem .6rem;border-radius:4px;font-size:.8rem;font-weight:500}
  .status-ok{background:#e6f4ea;color:#137333}
  .status-err{background:#fce8e6;color:#c5221f}
</style>
</head>
<body>
<h1>Contract Review Environment</h1>
<p class="sub">Interactive test UI — connect, reset, and step through a contract review episode.</p>

<div class="grid">
<div class="card">
  <h3 style="margin-top:0">Controls</h3>

  <label>Task</label>
  <select id="task">
    <option value="task1">Task 1 — Clause Extraction (Easy)</option>
    <option value="task2">Task 2 — Risk Identification (Medium)</option>
    <option value="task3">Task 3 — Full Contract Redline (Hard)</option>
  </select>

  <button class="btn-primary" onclick="doReset()">Reset Episode</button>
  <hr style="margin:1rem 0">

  <label>Action Type</label>
  <select id="action_type">
    <option value="extract_clause">extract_clause</option>
    <option value="flag_risk">flag_risk</option>
    <option value="annotate">annotate</option>
    <option value="approve_section">approve_section</option>
    <option value="request_revision">request_revision</option>
    <option value="submit_review">submit_review</option>
  </select>

  <label>Target Section</label>
  <input id="section" placeholder="e.g. 2.1" />

  <label>Content (clause type or risk type)</label>
  <input id="content" placeholder="e.g. confidentiality" />

  <label>Severity (for flag_risk)</label>
  <select id="severity">
    <option value="">-- none --</option>
    <option value="low">low</option>
    <option value="medium">medium</option>
    <option value="high">high</option>
    <option value="blocking">blocking</option>
  </select>

  <button class="btn-primary" onclick="doStep()" style="width:100%">Send Action →</button>
</div>

<div class="card">
  <h3 style="margin-top:0">Response</h3>
  <pre id="out">Click "Reset Episode" to start.</pre>
</div>
</div>

<script>
async function doReset() {
  const task = document.getElementById('task').value;
  const r = await fetch('/reset', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({task_id: task})
  });
  const data = await r.json();
  document.getElementById('out').textContent = JSON.stringify(data, null, 2);
}

async function doStep() {
  const sev = document.getElementById('severity').value;
  const action = {
    action_type: document.getElementById('action_type').value,
    target_section: document.getElementById('section').value,
    content: document.getElementById('content').value,
    severity: sev || null,
  };
  const r = await fetch('/step', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action})
  });
  const data = await r.json();
  document.getElementById('out').textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
"""
