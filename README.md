# AIM-Env: AI Email Triage RL Environment

AIM-Env is a lightweight, OpenEnv-compliant Reinforcement Learning execution environment where an LLM agent triages a simulated email inbox. It is designed to run as a single Python script inside a minimal Docker container and pass the Round 1 automated grader with a perfect score.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Environment Variables](#environment-variables)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
  - [Episode Lifecycle](#episode-lifecycle)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Reward Structure](#reward-structure)
  - [Grader & Scoring](#grader--scoring)
- [Task Configurations](#task-configurations)
- [Stdout Format Contract](#stdout-format-contract)
- [Data Models](#data-models)
- [Docker](#docker)
- [OpenEnv Spec Compliance](#openenv-spec-compliance)
- [Cleanup Checklist](#cleanup-checklist)

---

## Overview

The agent receives a simulated inbox of emails and must triage them by opening, classifying, prioritizing, routing, and detecting phishing — all within a fixed time budget. Each episode is driven by an LLM (via the OpenAI-compatible API) that reads the current inbox state and returns a JSON action.

The environment is fully deterministic: given the same seed, the same emails are always generated in the same order. This makes scoring reproducible across runs.

---

## Architecture

```
Grader / CI Runner
       │
       ▼  docker run
  inference.py  ◄──── HF_TOKEN / API_BASE_URL / MODEL_NAME (env vars)
       │
       ├── openai.OpenAI client  ──► LLM API
       │
       └── env/ module
             ├── AIMEnv          (reset / step / state / get_result)
             ├── EmailGenerator  (deterministic email synthesis)
             ├── Grader          (weighted scoring formula)
             └── Pydantic Models (Observation, Action, Reward, TaskConfig, ...)
```

The execution is entirely linear — a single Python process, no web server, no database, no frontend.

---

## Repository Structure

```
.
├── inference.py          # Entry point — the only file the grader executes
├── Dockerfile            # Lean python:3.10-slim image
├── requirements.txt      # openai + pydantic only
├── openenv.yaml          # OpenEnv task registry (easy / medium / hard)
│
├── env/                  # Core RL environment package
│   ├── __init__.py       # Exports all public symbols
│   ├── env.py            # AIMEnv class (reset, step, state, get_result)
│   ├── models.py         # All Pydantic models
│   ├── reward.py         # Reward model
│   ├── grader.py         # Deterministic scoring
│   └── email_generator.py# Seeded email synthesis
│
└── tasks/                # Standalone task config modules
    ├── task_easy.py
    ├── task_medium.py
    └── task_hard.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | **Yes** | — | API key passed to the OpenAI client. Raises `ValueError` immediately if missing. |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | Base URL for the OpenAI-compatible API endpoint. |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model identifier used for all LLM calls. |

`HF_TOKEN` is mandatory. The script will raise a `ValueError` and exit before doing any work if it is not set.

---

## Quick Start

### Run locally

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://api.openai.com/v1   # optional
export MODEL_NAME=gpt-4o-mini                   # optional

pip install openai pydantic
python inference.py
```

### Run with Docker

```bash
docker build -t aim-env .

docker run --rm \
  -e HF_TOKEN=your_token_here \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  aim-env
```

---

## How It Works

### Episode Lifecycle

For each of the three tasks (easy → medium → hard), `inference.py` runs one full episode:

1. `AIMEnv(config)` is instantiated with the task's `TaskConfig`.
2. `env.reset()` generates a fresh inbox of emails using the fixed seed and returns the initial `Observation`.
3. `[START]` is printed to stdout.
4. The `while not done` loop runs:
   - The current `Observation` is serialized into a natural-language prompt.
   - The LLM is called via `client.chat.completions.create(...)`.
   - The response is parsed as a JSON `Action`. If parsing fails for any reason, the fallback action is `{"type": "submit"}`.
   - `env.step(action)` is called, returning `(Observation, Reward, done)`.
   - `[STEP]` is printed to stdout.
5. After the loop, `env.get_result()` computes the `EpisodeResult` and `Grader().grade_episode(result)` produces the final score.
6. `[END]` is printed to stdout.

### Action Space

The agent can issue one of four action types per step:

| Action | Required Fields | Effect |
|---|---|---|
| `open` | `email_id` | Reveals the full email body; marks email as `opened`. Required before classifying. |
| `classify` | `email_id`, `category`, `priority`, `route` | Assigns category/priority/route to an opened email; marks it `processed`. |
| `detect_phishing` | `email_id` | Flags an email as phishing; marks it `processed`. |
| `submit` | — | Ends the episode immediately. |

The LLM must respond with a single JSON object, for example:

```json
{"type": "open", "email_id": "email_001"}
{"type": "classify", "email_id": "email_001", "category": "urgent", "priority": "high", "route": "escalate"}
{"type": "detect_phishing", "email_id": "email_002"}
{"type": "submit"}
```

### Observation Space

At each step the agent receives an `Observation` containing:

| Field | Type | Description |
|---|---|---|
| `inbox` | `List[EmailPartial]` | Up to 5 unread emails (id, subject, sender, preview) |
| `opened` | `List[str]` | IDs of currently opened emails |
| `time_left` | `int` | Remaining time units before timeout |
| `step_count` | `int` | Number of steps taken so far |
| `pending_emails` | `int` | Emails not yet processed |
| `alerts` | `List[str]` | Environment-level warnings |
| `classified` | `int` | Running count of correct classifications |
| `prioritized` | `int` | Running count of correct priorities |
| `routed` | `int` | Running count of correct routes |

### Reward Structure

Each step returns a `Reward` with a scalar `value` and a `components` dict breaking down the signal sources:

| Component | Value | Trigger |
|---|---|---|
| `step` | `-0.02` | Every step (time cost) |
| `opened` | `+0.01` | Successfully opening a new email |
| `correct_classification` | `+0.35` | Category matches ground truth |
| `correct_priority` | `+0.20` | Priority matches ground truth (requires correct category) |
| `correct_route` | `+0.20` | Route matches ground truth (requires correct category) |
| `wrong_classification` | `-0.25` | Category does not match ground truth |
| `wrong_priority` | `-0.10` | Priority wrong (category was correct) |
| `wrong_route` | `-0.10` | Route wrong (category was correct) |
| `phishing_detected` | `+0.60` | Correctly flagged phishing email |
| `phishing_detected_critical` | `+1.00` | Correctly flagged critical phishing email |
| `false_positive` | `-0.20` | Flagged a non-phishing email as phishing |
| `no_open_penalty` | `-0.15` | Tried to classify without opening first |
| `already_processed` | `-0.05` | Acted on an already-processed email |
| `invalid_action` | `-0.05` | Malformed or unknown action |
| `timeout` | `-0.50` | Time budget exhausted |
| `delay_<id>` | `-0.30` | Delayed penalty for ignoring a critical email |

### Grader & Scoring

The `Grader` computes a deterministic final score from the `EpisodeResult` using a weighted formula:

```
score = 0.30 × classification_acc
      + 0.20 × priority_acc
      + 0.20 × routing_acc
      + 0.20 × risk_score
      + 0.10 × efficiency_score
```

All component values are normalized to `[0.0, 1.0]`. The final score is clamped to `[0.0, 1.0]`.

An episode is considered a **success** if `score >= 0.5`.

---

## Task Configurations

Three tasks are defined with fixed seeds for full reproducibility:

| Task | Seed | Emails | Time Budget | Ambiguity | Phishing | Time Pressure |
|---|---|---|---|---|---|---|
| `easy` | 42 | 3 | 20 | 0.0 | No | 0.0 |
| `medium` | 137 | 7 | 30 | 0.2 | Yes | 0.1 |
| `hard` | 999 | 12 | 40 | 0.5 | Yes | 0.5 |

- **Ambiguity level**: probability that email context fields are partially missing, making classification harder.
- **Has phishing**: whether adversarial phishing emails are injected into the inbox.
- **Time pressure**: multiplier applied to step cost — higher values drain the time budget faster.

---

## Stdout Format Contract

The grader parses stdout line-by-line. Every line must match exactly:

### `[START]`
```
[START] task=<task_name> env=<benchmark> model=<model_name>
```
Example:
```
[START] task=easy env=aim-email-triage model=gpt-4o-mini
```

### `[STEP]`
```
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
```
- `reward` is always formatted to **exactly 2 decimal places**
- `done` is always lowercase `true` or `false`
- `error` is the literal string `null` when no error occurred, or the exception message string when an LLM/parse error happened
- `action_str` is `<type>` or `<type>:<email_id>` for actions targeting an email

Example:
```
[STEP] step=0 action=open:email_001 reward=-0.01 done=false error=null
[STEP] step=1 action=classify:email_001 reward=0.73 done=false error=null
[STEP] step=2 action=submit reward=-0.02 done=true error=null
```

### `[END]`
```
[END] success=<true|false> steps=<n> rewards=<r1,r2,...rn>
```
- `success` is lowercase `true` or `false`
- `rewards` is a comma-separated list of all per-step reward values, each to **exactly 2 decimal places**, no spaces

Example:
```
[END] success=true steps=3 rewards=-0.01,0.73,-0.02
```

---

## Data Models

All models are Pydantic v2 and live in `env/models.py`.

### `TaskConfig`
```python
class TaskConfig(BaseModel):
    num_emails: int        # number of emails to generate
    time_budget: int       # step budget before timeout
    seed: int              # deterministic RNG seed
    ambiguity_level: float # 0.0–1.0 noise on email context
    has_phishing: bool     # inject phishing emails
    time_pressure: float   # step cost multiplier
```

### `Observation`
```python
class Observation(BaseModel):
    inbox: List[EmailPartial]  # visible unread emails (max 5)
    opened: List[str]          # currently opened email IDs
    time_left: int
    step_count: int
    pending_emails: int
    alerts: List[str]
    classified: int
    prioritized: int
    routed: int
```

### `Action`
```python
class Action(BaseModel):
    type: str                        # open | classify | detect_phishing | submit
    email_id: Optional[str]
    category: Optional[EmailCategory]  # urgent|normal|spam|promotions|social|updates|forums
    priority: Optional[PriorityLevel]  # low|medium|high|critical
    route: Optional[RouteOption]       # inbox|archive|trash|escalate|review
```

### `Reward`
```python
class Reward(BaseModel):
    value: float                  # scalar step reward
    components: Dict[str, float]  # breakdown by cause
```

### `EpisodeResult`
```python
class EpisodeResult(BaseModel):
    score: float
    steps: int
    correct_classifications: int
    phishing_detected: int
    efficiency: float
    classification_acc: float   # correct_classifications / total_emails
    priority_acc: float         # correct_priorities / total_emails
    routing_acc: float          # correct_routes / total_emails
    risk_score: float           # phishing_detected / total_phishing
    efficiency_score: float     # 1 - (steps / (time_budget * 2))
```

---

## Docker

The `Dockerfile` is intentionally minimal:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "inference.py"]
```

`requirements.txt`:
```
openai
pydantic
```

No Node.js, no npm, no React, no FastAPI, no build tools. The image installs only what is needed to run `inference.py`.

Resource footprint is well within the Hugging Face container limits of 2 vCPU / 8 GB RAM.

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
| `step()`, `reset()`, `state()` API functions | ✅ |
| Minimum 3 tasks (Easy / Medium / Hard) | ✅ |
| Deterministic graders returning `[0.0, 1.0]` | ✅ |
| `openenv.yaml` config at repo root | ✅ |
| `python:3.10-slim` Docker base image | ✅ |

---

## Cleanup Checklist

Before submitting, delete the following directories and files that are not needed by the grader:

```bash
# Frontend (React / Vite)
rm -rf src/
rm -rf dist/
rm -f index.html vite.config.js postcss.config.js tailwind.config.js
rm -f package.json package-lock.json

# Backend (FastAPI)
rm -rf backend/

# Docker Compose (not needed for single-container grader)
rm -f docker-compose.yml

# Node modules (if present)
rm -rf node_modules/

# Training artifacts (not part of grader evaluation)
rm -rf training/
```

After cleanup, the only files needed are:

```
inference.py
Dockerfile
requirements.txt
openenv.yaml
env/
tasks/
```
