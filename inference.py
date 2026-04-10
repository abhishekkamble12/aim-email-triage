"""Inference module for the OpenEnv RL Email Triage project.

Wraps the RL inference loop in a FastAPI server (port 7860) so the
Hugging Face Space stays 'Running' after tasks complete. The inference
worker runs in a background thread on startup.

Stdout contract (parsed by the OpenEnv grader):
  [START] task=<name> env=<env> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel

from env import AIMEnv, Action, Grader
from env.models import Observation, TaskConfig


# ---------------------------------------------------------------------------
# Logging — write to stdout so the grader captures every line.
# propagate=True so pytest's caplog can also capture records in tests.
# ---------------------------------------------------------------------------

logger = logging.getLogger("inference")


def _configure_logger() -> None:
    """Attach a stdout StreamHandler if one isn't already present."""
    if not any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
        for h in logger.handlers
    ):
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Circuit-breaker state
# ---------------------------------------------------------------------------

class _CBState(Enum):
    CLOSED = "closed"      # normal operation
    OPEN = "open"          # tripped — skip LLM calls for this task


@dataclass
class _CircuitBreaker:
    """Per-task circuit breaker that trips on unrecoverable HTTP errors."""

    state: _CBState = _CBState.CLOSED

    def trip(self) -> None:
        self.state = _CBState.OPEN

    @property
    def is_open(self) -> bool:
        return self.state is _CBState.OPEN


# ---------------------------------------------------------------------------
# Stdout format helpers
# ---------------------------------------------------------------------------


def format_start(task_name: str, env_name: str, model_name: str) -> str:
    """Return the [START] stdout line for the given task/env/model names."""
    return f"[START] task={task_name} env={env_name} model={model_name}"


# ---------------------------------------------------------------------------
# EnvConfig
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    """Runtime configuration loaded from environment variables.

    Attributes:
        hf_token: HuggingFace API token used as the OpenAI-compatible API key.
        api_base_url: Base URL for the OpenAI-compatible inference endpoint.
        model_name: Name of the model to use for chat completions.
        timeout: Request timeout in seconds for each LLM API call.
    """

    hf_token: str
    api_base_url: str
    model_name: str
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "EnvConfig":
        """Construct an EnvConfig by reading values from os.environ.

        Provider swap guide — set these env vars to migrate away from HF:
          Groq:       API_BASE_URL=https://api.groq.com/openai/v1
                      HF_TOKEN=<groq_api_key>
                      MODEL_NAME=llama3-8b-8192
          OpenRouter: API_BASE_URL=https://openrouter.ai/api/v1
                      HF_TOKEN=<openrouter_api_key>
                      MODEL_NAME=mistralai/mistral-7b-instruct
          Together:   API_BASE_URL=https://api.together.xyz/v1
                      HF_TOKEN=<together_api_key>
                      MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2

        Returns:
            A fully populated EnvConfig instance.

        Raises:
            ValueError: If the required HF_TOKEN environment variable is absent.
        """
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is required")
        return cls(
            hf_token=hf_token,
            api_base_url=os.environ.get(
                "API_BASE_URL",
                "https://router.huggingface.co/novita/v3/openai",  # HF default
            ),
            model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
            timeout=int(os.environ.get("INFERENCE_TIMEOUT", "30")),
        )


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Stateless builder that converts an Observation into an LLM prompt string."""

    def build(self, obs: Observation, action_history: list[str] | None = None) -> str:
        """Build a prompt string from the given observation.

        Args:
            obs: The current environment observation.
            action_history: Ordered list of action strings already taken this
                episode. Injected into the prompt so the LLM avoids repeating
                itself (the root cause of the classify-loop bug).

        Returns:
            A formatted prompt string ready to send to the LLM.
        """
        if obs.inbox:
            inbox_section = "\n".join(
                f"  - id={e.id} subject={e.subject!r} sender={e.sender}"
                f" preview={e.preview!r}"
                for e in obs.inbox
            )
        else:
            inbox_section = "  (empty)"

        # [LOOP BREAKER] Show the last 5 actions so the model knows what to avoid
        if action_history:
            recent = action_history[-5:]
            history_section = (
                "\nActions already taken this episode (DO NOT repeat these):\n"
                + "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent))
                + "\n"
            )
        else:
            history_section = ""

        return (
            "You are an email triage agent.\n\n"
            f"Inbox:\n{inbox_section}\n"
            f"{history_section}\n"
            f"Time left: {obs.time_left}  Step: {obs.step_count}\n\n"
            "Available actions:\n"
            '  {"type": "open", "email_id": "<id>"}\n'
            '  {"type": "classify", "email_id": "<id>", "category": '
            '"<urgent|normal|spam|promotions|social|updates|forums>", '
            '"priority": "<low|medium|high|critical>", '
            '"route": "<inbox|archive|trash|escalate|review>"}\n'
            '  {"type": "detect_phishing", "email_id": "<id>"}\n'
            '  {"type": "submit"}\n\n'
            "Respond with a single JSON object representing your chosen action."
        )


# ---------------------------------------------------------------------------
# ActionParser
# ---------------------------------------------------------------------------


class ActionParser:
    """Parses raw LLM text output into a validated Action instance.

    Handles markdown-fenced JSON blocks and wraps parse errors as ValueError.
    """

    def parse(self, raw: str) -> Action:
        """Parse a raw LLM response string into an Action.

        Args:
            raw: The raw text returned by the LLM, possibly wrapped in
                markdown fences.

        Returns:
            A validated Action instance.

        Raises:
            ValueError: If the string is not valid JSON or cannot be used to
                construct an Action.
        """
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1)

        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise ValueError(str(exc)) from exc

        try:
            return Action(**parsed)
        except Exception as exc:
            raise ValueError(str(exc)) from exc


# ---------------------------------------------------------------------------
# EpisodeSummary
# ---------------------------------------------------------------------------


@dataclass
class EpisodeSummary:
    """Per-episode metrics returned by EpisodeRunner.run().

    Attributes:
        task_name: Human-readable task identifier (e.g. "easy").
        env_name: Environment identifier (e.g. "aim-email-triage").
        steps: Total number of steps executed in the episode.
        rewards: Reward value collected at each step.
        success: True when the graded score is >= 0.5.
    """

    task_name: str
    env_name: str
    steps: int
    rewards: list[float]
    success: bool


# ---------------------------------------------------------------------------
# EpisodeRunner
# ---------------------------------------------------------------------------


def _is_fatal_http_error(exc: Exception) -> bool:
    """Return True for HTTP 402/403/429 errors that should trip the circuit breaker."""
    msg = str(exc).lower()
    for marker in ("402", "403", "429", "payment required", "forbidden",
                   "permission", "too many requests", "rate limit", "depleted"):
        if marker in msg:
            return True
    status = getattr(exc, "status_code", None)
    if status in (402, 403, 429):
        return True
    return False


class EpisodeRunner:
    """Executes a single RL episode loop (reset → step → grade)."""

    def __init__(
        self,
        env: AIMEnv,
        client: OpenAI,
        config: EnvConfig,
        circuit_breaker: _CircuitBreaker | None = None,
        loop_break_threshold: int = 3,
    ) -> None:
        self.env = env
        self.client = client
        self.config = config
        self._prompt_builder = PromptBuilder()
        self._action_parser = ActionParser()
        self._cb = circuit_breaker or _CircuitBreaker()
        # Loop-breaker: tracks consecutive identical actions to prevent credit drain
        self._loop_break_threshold = loop_break_threshold
        self._last_action_str: str | None = None
        self._consecutive_count: int = 0

    def run(self, task_name: str, env_name: str) -> EpisodeSummary:
        """Execute one full episode and return a summary."""
        obs = self.env.reset()
        done = False
        step_num = 0
        rewards_list: list[float] = []
        # History fed back into the prompt so the LLM knows what it already tried
        action_history: list[str] = []

        while not done:
            error: str | None = None
            action: Action

            # --- Circuit breaker: skip LLM if tripped, submit immediately ---
            if self._cb.is_open:
                error = "circuit_breaker_open"
                action = Action(type="submit")
            else:
                # [LOOP BREAKER] Inject history so the model sees prior actions
                prompt = self._prompt_builder.build(obs, action_history)
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        timeout=self.config.timeout,
                    )
                    raw = (response.choices[0].message.content or "").strip()
                    action = self._action_parser.parse(raw)

                except (TimeoutError, ConnectionError) as exc:
                    logger.warning("LLM transient error: %s", exc)
                    error = str(exc)
                    action = Action(type="submit")

                except Exception as exc:
                    error = str(exc)
                    if _is_fatal_http_error(exc):
                        logger.error(
                            "Circuit breaker tripped (fatal HTTP error): %s", exc
                        )
                        self._cb.trip()
                        # [FIX] Terminate immediately — don't burn another API call
                        done = True
                        rewards_list.append(-1.0)
                        logger.info(
                            "[STEP] step=%d action=submit reward=-1.00"
                            " done=true error=%s",
                            step_num, error,
                        )
                        step_num += 1
                        break
                    action = Action(type="submit")

            # Build canonical action string for loop detection and logging
            action_str = action.type
            if action.email_id:
                action_str += f":{action.email_id}"

            # [LOOP BREAKER] Force submit if same action repeated N times in a row
            if action_str == self._last_action_str:
                self._consecutive_count += 1
                if self._consecutive_count >= self._loop_break_threshold:
                    logger.warning(
                        "Loop breaker triggered after %d consecutive '%s' actions"
                        " — forcing submit",
                        self._consecutive_count, action_str,
                    )
                    action = Action(type="submit")
                    action_str = "submit"
                    error = f"loop_breaker:{self._consecutive_count}"
                    self._consecutive_count = 0
            else:
                self._last_action_str = action_str
                self._consecutive_count = 1

            action_history.append(action_str)

            try:
                obs, reward, done = self.env.step(action)
            except Exception as exc:
                logger.error("env.step raised: %s", exc)
                done = True
                continue

            rewards_list.append(reward.value)

            logger.info(
                "[STEP] step=%d action=%s reward=%.2f done=%s error=%s",
                step_num,
                action_str,
                reward.value,
                str(done).lower(),
                error if error is not None else "null",
            )
            step_num += 1

        result = self.env.get_result()
        score = Grader().grade_episode(result)  # type: ignore[no-untyped-call]
        return EpisodeSummary(
            task_name=task_name,
            env_name=env_name,
            steps=step_num,
            rewards=rewards_list,
            success=score >= 0.5,
        )


# ---------------------------------------------------------------------------
# InferenceRunner
# ---------------------------------------------------------------------------


class InferenceRunner:
    """Top-level orchestrator that runs all tasks end-to-end."""

    DEFAULT_TASKS: list[tuple[str, TaskConfig]] = [
        (
            "easy",
            TaskConfig(
                seed=42,
                num_emails=3,
                time_budget=20,
                ambiguity_level=0.0,
                has_phishing=False,
                time_pressure=0.0,
            ),
        ),
        (
            "medium",
            TaskConfig(
                seed=137,
                num_emails=7,
                time_budget=30,
                ambiguity_level=0.2,
                has_phishing=True,
                time_pressure=0.1,
            ),
        ),
        (
            "hard",
            TaskConfig(
                seed=999,
                num_emails=12,
                time_budget=40,
                ambiguity_level=0.5,
                has_phishing=True,
                time_pressure=0.5,
            ),
        ),
    ]

    def __init__(
        self,
        config: EnvConfig | None = None,
        tasks: list[tuple[str, TaskConfig]] | None = None,
    ) -> None:
        _configure_logger()
        self.config = config if config is not None else EnvConfig.from_env()
        self.tasks = tasks if tasks is not None else self.DEFAULT_TASKS

    def run_all(self) -> None:
        """Run every task, emitting [START]/[END] logs and isolating failures."""
        client = OpenAI(base_url=self.config.api_base_url, api_key=self.config.hf_token)

        for task_name, task_config in self.tasks:
            env_name = "aim-email-triage"
            cb = _CircuitBreaker()  # fresh breaker per task

            # --- Exact stdout contract ---
            logger.info(
                "[START] task=%s env=%s model=%s",
                task_name,
                env_name,
                self.config.model_name,
            )
            try:
                env = AIMEnv(task_config)
                runner = EpisodeRunner(env, client, self.config, cb)
                summary = runner.run(task_name, env_name)
                rewards_str = ",".join(f"{r:.2f}" for r in summary.rewards)

                # --- Exact stdout contract ---
                logger.info(
                    "[END] success=%s steps=%d rewards=%s",
                    str(summary.success).lower(),
                    summary.steps,
                    rewards_str,
                )
            except Exception as exc:
                logger.error("Unhandled exception for task %s: %s", task_name, exc)


# ---------------------------------------------------------------------------
# FastAPI server — OpenEnv-compliant API mode
# The grader acts as the agent; we just listen and respond synchronously.
# Background inference loop is disabled in this mode.
# ---------------------------------------------------------------------------

# Global environment instance — maintains state between API calls
_env: AIMEnv | None = None

_DEFAULT_TASK_CONFIG = TaskConfig(
    seed=42,
    num_emails=5,
    time_budget=30,
    ambiguity_level=0.2,
    has_phishing=True,
    time_pressure=0.1,
)


class ResetRequest(BaseModel):
    seed: int | None = None


class StepRequest(BaseModel):
    action: dict | str


app = FastAPI(title="AIM-Env Inference", version="1.0.0")


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"service": "AIM-Env Inference", "status": "ready"})


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/reset")
def reset(body: ResetRequest | None = None) -> JSONResponse:
    """Reset the environment and return the initial observation."""
    global _env
    seed = (body.seed if body else None) or _DEFAULT_TASK_CONFIG.seed
    config = TaskConfig(
        seed=seed,
        num_emails=_DEFAULT_TASK_CONFIG.num_emails,
        time_budget=_DEFAULT_TASK_CONFIG.time_budget,
        ambiguity_level=_DEFAULT_TASK_CONFIG.ambiguity_level,
        has_phishing=_DEFAULT_TASK_CONFIG.has_phishing,
        time_pressure=_DEFAULT_TASK_CONFIG.time_pressure,
    )
    _env = AIMEnv(config)
    obs: Observation = _env.reset()
    return JSONResponse(obs.model_dump())


@app.post("/step")
def step(body: StepRequest) -> JSONResponse:
    """Advance the environment by one step."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    # Parse action — accept dict or JSON string
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
# Entry point
# ---------------------------------------------------------------------------


def run_inference() -> None:
    """Entry point for the standalone inference loop (no server).

    Called by the evaluator or CI pipeline to execute all tasks and emit
    the structured [START]/[STEP]/[END] stdout contract without starting
    any web server.
    """
    try:
        InferenceRunner().run_all()
    except Exception as exc:  # noqa: BLE001
        # Structured fallback so downstream Output Parsing can still execute.
        print(json.dumps({"status": "error", "message": str(exc), "data": None}))


# ---------------------------------------------------------------------------
# Entry point — ONLY runs when executed directly, never on import.
#
# The HF Space always needs a live server on 0.0.0.0:7860 or the container
# exits. Strategy:
#   1. If --serve is passed (or no flag): start server immediately.
#      Inference runs in a background thread so the server is always alive.
#   2. The evaluator POSTs to /reset and /step directly — the server handles it.
#   3. If the port is already bound (Errno 98), exit cleanly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import threading

    _base_port = int(os.environ.get("PORT", 7860))  # never hardcode 7860

    if "--serve" in sys.argv:
        # ----------------------------------------------------------------
        # SERVER MODE — only when explicitly requested (HF Space startup,
        # docker run, etc.).  Inference runs in a background daemon thread
        # so the FastAPI server stays alive indefinitely.
        # ----------------------------------------------------------------
        def _bg_inference() -> None:
            try:
                InferenceRunner().run_all()
            except Exception as exc:
                logger.error("Background inference failed: %s", exc)

        threading.Thread(target=_bg_inference, daemon=True).start()

        for _port in (_base_port, _base_port + 1, _base_port + 2):
            try:
                uvicorn.run(app, host="0.0.0.0", port=_port, log_level="warning")
                break
            except OSError as _exc:
                if getattr(_exc, "errno", None) == 98 or "[Errno 98]" in str(_exc):
                    # Port already locked — evaluator container detected.
                    print("Port in use, bypassing", file=sys.stderr)
                    sys.exit(0)
                print(f"WARNING: Could not bind to port {_port}: {_exc}", file=sys.stderr)
        else:
            print("ERROR: All ports exhausted, exiting gracefully.", file=sys.stderr)

    else:
        # ----------------------------------------------------------------
        # EVALUATOR / HEADLESS MODE — `python inference.py` with no flags.
        # The hackathon bot runs the script this way.  Never touch a port.
        # ----------------------------------------------------------------
        print("Evaluator mode detected", file=sys.stderr)
        run_inference()
