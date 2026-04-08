---
title: AIM-Env — AI Email Triage
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - llm
---

# AIM-Env: AI Email Triage RL Environment

AIM-Env is an [OpenEnv](https://openenv.dev)-compliant reinforcement learning environment where an LLM agent triages a simulated email inbox. An LLM-powered agent reads inbox state, decides actions (open, classify, detect phishing, submit), and is scored on accuracy, routing, and efficiency.

The project ships three components:

| Component | Purpose |
|---|---|
| `inference.py` | OpenEnv entry point — the only file the grader executes |
| `backend/` | FastAPI server for the interactive web demo |
| `src/` | React + Vite frontend dashboard |

---

## Table of Contents

- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
  - [Run inference (grader mode)](#run-inference-grader-mode)
  - [Run the web demo](#run-the-web-demo)
  - [Run with Docker](#run-with-docker)
- [Environment Variables](#environment-variables)
- [How the RL Environment Works](#how-the-rl-environment-works)
  - [Episode Lifecycle](#episode-lifecycle)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Reward Structure](#reward-structure)
  - [Grader & Scoring](#grader--scoring)
- [Task Configurations](#task-configurations)
- [Stdout Format Contract](#stdout-format-contract)
- [Code Architecture (inference.py)](#code-architecture-inferencepy)
- [Data Models](#data-models)
- [Training](#training)
- [Testing](#testing)
- [Docker](#docker)
- [OpenEnv Spec Compliance](#openenv-spec-compliance)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OpenEnv Grader / CI                  │
└────────────────────────┬────────────────────────────────┘
                         │ docker run
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    inference.py                         │
│                                                         │
│  InferenceRunner                                        │
│    └── EpisodeRunner (per task)                         │
│          ├── PromptBuilder  ──► LLM prompt              │
│          ├── ActionParser   ◄── LLM response            │
│          └── AIMEnv         ──► step / reset / grade    │
│                                                         │
│  Config: HF_TOKEN / API_BASE_URL / MODEL_NAME           │
└─────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────────────────┐
│  React Frontend  │────►│  FastAPI Backend              │
│  (src/)          │     │  (backend/)                   │
│  Vite + Tailwind │     │  /api/run-demo                │
│  Dashboard/Demo  │     │  /api/train-agent             │
└──────────────────┘     │  /api/metrics                 │
                         └──────────────────────────────┘
```

---

## Repository Structure

```
.
├── inference.py              # OpenEnv entry point (grader executes this)
├── Dockerfile                # python:3.10-slim image for inference.py
├── docker-compose.yml        # Runs backend + frontend together
├── requirements.txt          # openai, pydantic, fastapi, uvicorn
├── openenv.yaml              # OpenEnv task registry (easy/medium/hard)
├── test_inference.py         # Unit + property-based tests (pytest + Hypothesis)
│
├── env/                      # Core RL environment package
│   ├── __init__.py
│   ├── env.py                # AIMEnv — reset(), step(), state(), get_result()
│   ├── models.py             # Pydantic models: Observation, Action, Reward, ...
│   ├── reward.py             # Per-step reward computation
│   ├── grader.py             # Deterministic episode scoring
│   └── email_generator.py    # Seeded, reproducible email synthesis
│
├── tasks/                    # Task config modules
│   ├── task_easy.py
│   ├── task_medium.py
│   └── task_hard.py
│
├── backend/                  # FastAPI web API (demo/dashboard only)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py           # FastAPI app, CORS, rate limiting
│       ├── api/routes.py     # /run-demo, /run-task, /train-agent, /metrics
│       ├── services/
│       │   ├── env_service.py
│       │   ├── agent_service.py
│       │   └── training_service.py
│       └── schemas/          # Pydantic request/response schemas
│
├── src/                      # React + Vite frontend
│   ├── pages/
│   │   ├── Home.jsx          # Landing page with live stats
│   │   ├── Demo.jsx          # Interactive agent demo
│   │   ├── Dashboard.jsx     # Training metrics & analytics
│   │   ├── Architecture.jsx  # System diagram
│   │   └── About.jsx
│   ├── components/
│   │   ├── Navbar.jsx
│   │   └── Footer.jsx
│   └── services/api.js       # Axios client for backend
│
└── training/
    ├── train.py              # Q-learning + LLM hybrid training loop
    ├── scores.json           # Saved training scores
    └── learning_curve.png    # Training curve visualization
```

---

## Quick Start

### Run inference (grader mode)

```bash
pip install openai pydantic

export HF_TOKEN=your_token_here
export API_BASE_URL=https://api.openai.com/v1   # optional
export MODEL_NAME=gpt-4o-mini                   # optional

python inference.py
```

### Run the web demo

Requires Node.js 18+ and Python 3.10+.

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
npm install
npm run dev
```

Then open http://localhost:5173.

### Run with Docker

**Inference only (grader mode):**

```bash
docker build -t aim-env .
docker run --rm \
  -e HF_TOKEN=your_token_here \
  aim-env
```

**Full stack (backend + frontend):**

```bash
docker-compose up --build
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | **Yes** | — | API key for the LLM client. Raises `ValueError` immediately if missing. |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | Base URL for any OpenAI-compatible endpoint. |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier used for all completions. |
| `INFERENCE_TIMEOUT` | No | `30` | Per-request timeout in seconds. |

---

## How the RL Environment Works

### Episode Lifecycle

For each of the three tasks (easy → medium → hard), one full episode runs:

1. `AIMEnv(config)` is instantiated with the task's `TaskConfig`.
2. `env.reset()` generates a deterministic inbox using the fixed seed and returns the initial `Observation`.
3. `[START]` is logged to stdout.
4. The step loop runs until `done`:
   - `PromptBuilder` converts the `Observation` into a natural-language prompt.
   - The LLM is called via `client.chat.completions.create(...)`.
   - `ActionParser` extracts the JSON action from the response (handles markdown fences). On any parse failure, falls back to `{"type": "submit"}`.
   - `env.step(action)` returns `(Observation, Reward, done)`.
   - `[STEP]` is logged to stdout.
5. `env.get_result()` + `Grader().grade_episode()` compute the final score.
6. `[END]` is logged to stdout.

### Action Space

| Action | Required Fields | Effect |
|---|---|---|
| `open` | `email_id` | Reveals full email body; marks email as opened. Required before classifying. |
| `classify` | `email_id`, `category`, `priority`, `route` | Assigns category/priority/route to an opened email. |
| `detect_phishing` | `email_id` | Flags an email as phishing. |
| `submit` | — | Ends the episode immediately. |

Example LLM responses:

```json
{"type": "open", "email_id": "email_001"}
{"type": "classify", "email_id": "email_001", "category": "urgent", "priority": "high", "route": "escalate"}
{"type": "detect_phishing", "email_id": "email_002"}
{"type": "submit"}
```

### Observation Space

| Field | Type | Description |
|---|---|---|
| `inbox` | `List[EmailPartial]` | Up to 5 unread emails (id, subject, sender, preview) |
| `opened` | `List[str]` | IDs of currently opened emails |
| `time_left` | `int` | Remaining time units |
| `step_count` | `int` | Steps taken so far |
| `pending_emails` | `int` | Emails not yet processed |
| `alerts` | `List[str]` | Environment-level warnings |
| `classified` | `int` | Running count of correct classifications |
| `prioritized` | `int` | Running count of correct priorities |
| `routed` | `int` | Running count of correct routes |

### Reward Structure

| Component | Value | Trigger |
|---|---|---|
| `step` | `-0.02` | Every step (time cost) |
| `opened` | `+0.01` | Successfully opening a new email |
| `correct_classification` | `+0.35` | Category matches ground truth |
| `correct_priority` | `+0.20` | Priority correct (requires correct category) |
| `correct_route` | `+0.20` | Route correct (requires correct category) |
| `wrong_classification` | `-0.25` | Category mismatch |
| `wrong_priority` | `-0.10` | Priority wrong (category was correct) |
| `wrong_route` | `-0.10` | Route wrong (category was correct) |
| `phishing_detected` | `+0.60` | Correctly flagged phishing email |
| `phishing_detected_critical` | `+1.00` | Correctly flagged critical phishing |
| `false_positive` | `-0.20` | Flagged a non-phishing email as phishing |
| `no_open_penalty` | `-0.15` | Tried to classify without opening first |
| `already_processed` | `-0.05` | Acted on an already-processed email |
| `invalid_action` | `-0.05` | Malformed or unknown action |
| `timeout` | `-0.50` | Time budget exhausted |

### Grader & Scoring

```
score = 0.30 × classification_acc
      + 0.20 × priority_acc
      + 0.20 × routing_acc
      + 0.20 × risk_score
      + 0.10 × efficiency_score
```

All components are normalized to `[0.0, 1.0]`. Final score is clamped to `[0.0, 1.0]`.

An episode is a **success** when `score >= 0.5`.

---

## Task Configurations

| Task | Seed | Emails | Time Budget | Ambiguity | Phishing | Time Pressure |
|---|---|---|---|---|---|---|
| `easy` | 42 | 3 | 20 | 0.0 | No | 0.0 |
| `medium` | 137 | 7 | 30 | 0.2 | Yes | 0.1 |
| `hard` | 999 | 12 | 40 | 0.5 | Yes | 0.5 |

- **Ambiguity level**: probability that email context is partially missing, making classification harder.
- **Has phishing**: whether adversarial phishing emails are injected.
- **Time pressure**: multiplier on step cost — higher values drain the budget faster.

---

## Stdout Format Contract

The grader parses stdout line-by-line. Lines must match exactly:

### `[START]`
```
[START] task=<task_name> env=<env_name> model=<model_name>
```
```
[START] task=easy env=aim-email-triage model=gpt-4o-mini
```

### `[STEP]`
```
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
```
- `reward` — always 2 decimal places
- `done` — lowercase `true` or `false`
- `error` — literal `null` when no error, otherwise the exception message
- `action_str` — `<type>` or `<type>:<email_id>`

```
[STEP] step=0 action=open:email_001 reward=-0.01 done=false error=null
[STEP] step=1 action=classify:email_001 reward=0.73 done=false error=null
[STEP] step=2 action=submit reward=-0.02 done=true error=null
```

### `[END]`
```
[END] success=<true|false> steps=<n> rewards=<r1,r2,...>
```
- `rewards` — comma-separated, each to 2 decimal places, no spaces

```
[END] success=true steps=3 rewards=-0.01,0.73,-0.02
```

---

## Code Architecture (inference.py)

`inference.py` is structured in dependency order:

```
EnvConfig → PromptBuilder → ActionParser → EpisodeSummary → EpisodeRunner → InferenceRunner
```

| Class | Role |
|---|---|
| `EnvConfig` | Dataclass holding runtime config; `from_env()` reads env vars and raises `ValueError` if `HF_TOKEN` is missing |
| `PromptBuilder` | Stateless; converts an `Observation` into a natural-language LLM prompt |
| `ActionParser` | Parses raw LLM text (including markdown-fenced JSON) into a validated `Action`; wraps errors as `ValueError` |
| `EpisodeSummary` | Dataclass capturing per-episode metrics: task name, steps, rewards, success flag |
| `EpisodeRunner` | Runs one full episode loop; handles LLM timeouts, connection errors, and env step failures with graceful fallbacks |
| `InferenceRunner` | Top-level orchestrator; iterates over all tasks, emits `[START]`/`[END]` logs, isolates per-task failures |

Error handling summary:

| Scenario | Handler | Fallback |
|---|---|---|
| `HF_TOKEN` missing | `EnvConfig.from_env` | `ValueError` — process exits |
| LLM `TimeoutError` / `ConnectionError` | `EpisodeRunner` | `WARNING` log + `Action(type="submit")` |
| Any other LLM exception | `EpisodeRunner` | error recorded in `[STEP]` log + `Action(type="submit")` |
| `env.step` exception | `EpisodeRunner` | `ERROR` log + `done = True` |
| `EpisodeRunner.run` unhandled exception | `InferenceRunner.run_all` | `ERROR` log + continue to next task |

---

## Data Models

All models are Pydantic v2 and live in `env/models.py`.

```python
class TaskConfig(BaseModel):
    num_emails: int
    time_budget: int
    seed: int
    ambiguity_level: float
    has_phishing: bool
    time_pressure: float

class Observation(BaseModel):
    inbox: List[EmailPartial]
    opened: List[str]
    time_left: int
    step_count: int
    pending_emails: int
    alerts: List[str]
    classified: int
    prioritized: int
    routed: int

class Action(BaseModel):
    type: str                          # open | classify | detect_phishing | submit
    email_id: Optional[str]
    category: Optional[EmailCategory]  # urgent|normal|spam|promotions|social|updates|forums
    priority: Optional[PriorityLevel]  # low|medium|high|critical
    route: Optional[RouteOption]       # inbox|archive|trash|escalate|review

class Reward(BaseModel):
    value: float
    components: Dict[str, float]
```

---

## Training

`training/train.py` implements a Q-learning + LLM hybrid agent for offline experimentation:

- Epsilon-greedy exploration (ε starts at 0.9, decays to 0.1)
- Q-values stored per `(opened_emails, time_left)` state key
- Few-shot experience replay: top positive-reward steps are injected into the LLM prompt
- Runs 100 episodes across easy/medium/hard tasks

```bash
export OPENAI_API_KEY=your_key   # optional — runs in mock mode without it
python training/train.py
```

Results are saved to `training/scores.json` and visualized in `training/learning_curve.png`.

---

## Testing

Tests live in `test_inference.py` alongside `inference.py`. The suite covers all six classes with both unit tests and property-based tests via [Hypothesis](https://hypothesis.readthedocs.io/).

```bash
pip install pytest hypothesis pytest-mock
pytest test_inference.py -v
```

| Test type | Count | What it covers |
|---|---|---|
| Unit tests | 14 | `EnvConfig.from_env` defaults/errors, `PromptBuilder` rendering, `EpisodeRunner` call counts, `InferenceRunner` log format |
| Property tests | 10 | Field round-trips, `ActionParser` round-trip, markdown fence transparency, invalid JSON, prompt completeness, rewards length, LLM exception fallback, `run_all` task count |

All 24 tests pass in ~32 seconds.

---

## Docker

**Inference image** (`Dockerfile` at repo root):

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "inference.py"]
```

No Node.js, no build tools. Installs only `openai` and `pydantic`. Well within Hugging Face container limits (2 vCPU / 8 GB RAM).

**Full stack** (`docker-compose.yml`): spins up the FastAPI backend on port 8000 and the React frontend on port 3000.

---

## OpenEnv Spec Compliance

| Requirement | Status |
|---|---|
| Single `inference.py` entry point at repo root | ✅ |
| `HF_TOKEN` mandatory, `ValueError` if missing | ✅ |
| `API_BASE_URL` with default fallback | ✅ |
| `MODEL_NAME` with default fallback | ✅ |
| Official `openai.OpenAI` client only | ✅ |
| `[START]` / `[STEP]` / `[END]` stdout format | ✅ |
| Rewards to exactly 2 decimal places | ✅ |
| Booleans as lowercase strings | ✅ |
| `error=null` literal when no error | ✅ |
| Typed Pydantic `Observation`, `Action`, `Reward` models | ✅ |
| `step()`, `reset()`, `state()`, `get_result()` API | ✅ |
| Minimum 3 tasks (easy / medium / hard) | ✅ |
| Deterministic grader returning `[0.0, 1.0]` | ✅ |
| `openenv.yaml` config at repo root | ✅ |
| `python:3.10-slim` Docker base image | ✅ |
| `mypy --strict` clean | ✅ |
| `ruff check` clean | ✅ |
