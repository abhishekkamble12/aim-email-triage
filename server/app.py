"""OpenEnv-compliant FastAPI server for the AIM Email Triage environment.

Endpoints:
  POST /reset  — initialise/reset the environment, returns initial Observation
  POST /step   — advance one step, returns {observation, reward, done, info}
"""

from __future__ import annotations

import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import AIMEnv, Action
from env.models import Observation, TaskConfig

# ---------------------------------------------------------------------------
# Global env instance — maintains state between API calls
# ---------------------------------------------------------------------------

_env: AIMEnv | None = None

_DEFAULT_CONFIG = TaskConfig(
    seed=42,
    num_emails=5,
    time_budget=30,
    ambiguity_level=0.2,
    has_phishing=True,
    time_pressure=0.1,
)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: int | None = None


class StepRequest(BaseModel):
    action: dict | str

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AIM-Env OpenEnv Server", version="1.0.0")


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"service": "AIM-Env OpenEnv Server", "status": "ready"})


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/reset")
def reset(body: ResetRequest | None = None) -> JSONResponse:
    """Reset the environment. Accepts optional {"seed": int}."""
    global _env
    seed = (body.seed if body else None) or _DEFAULT_CONFIG.seed
    config = TaskConfig(
        seed=seed,
        num_emails=_DEFAULT_CONFIG.num_emails,
        time_budget=_DEFAULT_CONFIG.time_budget,
        ambiguity_level=_DEFAULT_CONFIG.ambiguity_level,
        has_phishing=_DEFAULT_CONFIG.has_phishing,
        time_pressure=_DEFAULT_CONFIG.time_pressure,
    )
    _env = AIMEnv(config)
    obs: Observation = _env.reset()
    return JSONResponse(obs.model_dump())


@app.post("/step")
def step(body: StepRequest) -> JSONResponse:
    """Advance the environment by one step."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised — call /reset first.")

    raw_action = body.action
    if isinstance(raw_action, str):
        try:
            raw_action = json.loads(raw_action)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid action JSON: {exc}")

    try:
        action = Action(**raw_action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action fields: {exc}")

    try:
        obs, reward, done = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return JSONResponse({
        "observation": obs.model_dump(),
        "reward": round(reward.value, 2),
        "done": done,
        "info": reward.components,
    })


# ---------------------------------------------------------------------------
# Entry point (used by [project.scripts] server = "server.app:main")
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the server on port 7860 (HF Spaces default)."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="warning")


if __name__ == "__main__":
    main()
