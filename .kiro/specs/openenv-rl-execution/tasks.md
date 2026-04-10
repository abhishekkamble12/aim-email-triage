# Implementation Plan: openenv-rl-execution

## Overview

Refactor the AIM-Env Platform into a strictly compliant OpenEnv RL execution environment. The implementation replaces the existing `inference.py` and root `Dockerfile`, adds `openenv.yaml`, adds a `state()` accessor to `AIMEnv`, and adds a lean `requirements.txt` — all while leaving the `env/` module otherwise unchanged.

## Tasks

- [x] 1. Add `state()` accessor to `AIMEnv` and verify env module compliance
  - Add `def state(self) -> Observation: return self.current_obs` to `env/env.py`
  - Confirm `Observation`, `Action`, `Reward`, `TaskConfig`, `EpisodeResult`, `AIMEnv`, and `Grader` are all importable from `env/`
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 2. Rewrite `inference.py` as the OpenEnv-compliant entry point
  - [x] 2.1 Implement `get_env_config()` — read `HF_TOKEN` (raise `ValueError` if absent), `API_BASE_URL` (default `"https://api.openai.com/v1"`), `MODEL_NAME` (default `"gpt-4o-mini"`)
    - _Requirements: 1.1, 1.2, 1.3_

  - [x]* 2.2 Write property test for `get_env_config()` — Property 1: Missing HF_TOKEN raises ValueError
    - **Property 1: Missing HF_TOKEN raises ValueError**
    - **Validates: Requirements 1.1**
    - Use `@given(st.none())`, monkeypatch env to remove `HF_TOKEN`, assert `ValueError`

  - [x] 2.3 Implement `format_start(task_name, env_name, model_name) -> str`
    - Return `[START] task=<task_name> env=<env_name> model=<model_name>`
    - _Requirements: 2.1_

  - [x]* 2.4 Write property test for `format_start()` — Property 2: [START] line contains all required fields
    - **Property 2: [START] line contains all required fields**
    - **Validates: Requirements 2.1**
    - Use `@given(st.text(min_size=1), st.text(min_size=1), st.text(min_size=1))`

  - [x] 2.5 Implement `format_step(step, action, reward, done, error) -> str`
    - Return `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
    - Reward: 2 decimal places; booleans: lowercase; error: literal `"null"` when `None`
    - _Requirements: 2.2, 2.4, 2.5_

  - [ ]* 2.6 Write property test for `format_step()` — Property 3: [STEP] line correct formatting
    - **Property 3: [STEP] line contains all required fields with correct formatting**
    - **Validates: Requirements 2.2, 2.4, 2.5**
    - Use `@given(st.integers(min_value=0), st.text(min_size=1), st.floats(allow_nan=False, allow_infinity=False), st.booleans(), st.one_of(st.none(), st.text(min_size=1)))`

  - [x] 2.7 Implement `format_end(success, steps, rewards) -> str`
    - Return `[END] success=<true|false> steps=<n> rewards=<r1,r2,...rn>` with each reward to 2 decimal places
    - _Requirements: 2.3, 2.4, 2.5_

  - [ ]* 2.8 Write property test for `format_end()` — Property 4: [END] line correct formatting
    - **Property 4: [END] line contains all required fields**
    - **Validates: Requirements 2.3, 2.4, 2.5**
    - Use `@given(st.booleans(), st.integers(min_value=0), st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False)))`

  - [x] 2.9 Implement `run_episode(env, client, model, task_name, env_name) -> None`
    - Call `env.reset()`, run `while not done` loop, call LLM via `openai.OpenAI` client
    - Print `[START]`, one `[STEP]` per step, then `[END]` using the format functions
    - On LLM failure: catch exception, set `error` field in `[STEP]`, fall back to `Action(type="submit")`
    - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 6.1_

  - [ ]* 2.10 Write property test for episode termination — Property 7: Episode loop always terminates
    - **Property 7: Episode loop always terminates with done=True**
    - **Validates: Requirements 6.1**
    - Use `@given(st.sampled_from([EASY_TASK_CONFIG, MEDIUM_TASK_CONFIG, HARD_TASK_CONFIG]))`, run with heuristic agent, assert `done == True` and `steps > 0`

  - [x] 2.11 Implement `main()` — wire `get_env_config()`, construct `openai.OpenAI` client, iterate over Easy/Medium/Hard tasks, call `run_episode()` for each
    - _Requirements: 1.1, 1.2, 1.3, 4.1, 4.2_

- [x] 3. Checkpoint — Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement and verify the deterministic `Grader`
  - [x] 4.1 Confirm `Grader.grade_episode()` uses the weighted formula: `0.30*cls + 0.20*pri + 0.20*rte + 0.20*risk + 0.10*eff` and clamps to `[0.0, 1.0]`
    - No code changes needed if already correct; add a docstring clarifying the formula
    - _Requirements: 3.4_

  - [ ]* 4.2 Write property test for `Grader` — Property 5: Grader always returns score in [0.0, 1.0]
    - **Property 5: Grader always returns a score in [0.0, 1.0]**
    - **Validates: Requirements 3.4**
    - Use `@given(st.floats(0, 1), st.floats(0, 1), st.floats(0, 1), st.floats(0, 1), st.floats(0, 1))`, build `EpisodeResult`, assert `0.0 <= score <= 1.0`

  - [ ]* 4.3 Write property test for `AIMEnv.reset()` — Property 6: reset() returns valid Observation
    - **Property 6: reset() always returns a valid Observation**
    - **Validates: Requirements 3.1**
    - Use `@given(st.integers(1, 15), st.integers(10, 60), st.integers(0, 9999))`, build `TaskConfig`, call `reset()`, assert `time_left == config.time_budget`, `step_count == 0`, `done == False`

- [x] 5. Create `openenv.yaml`
  - Write `openenv.yaml` at the repo root with `name: aim-email-triage`, `version: "1.0"`, and three tasks: easy (seed=42, num_emails=3, time_budget=20), medium (seed=137, num_emails=7, time_budget=30), hard (seed=999, num_emails=12, time_budget=40)
  - _Requirements: 7.1, 7.2_

- [x] 6. Create lean `Dockerfile` and `requirements.txt`
  - Replace root `Dockerfile` with `FROM python:3.10-slim`, `WORKDIR /app`, `COPY requirements.txt .`, `RUN pip install --no-cache-dir -r requirements.txt`, `COPY . .`, `CMD ["python", "inference.py"]`
  - Write `requirements.txt` at repo root containing only `openai` and `pydantic`
  - _Requirements: 8.1, 8.2, 8.3_

- [ ]* 6.1 Write smoke tests for Dockerfile and openenv.yaml
  - Verify `Dockerfile` `FROM` line is `python:3.10-slim`
  - Verify `requirements.txt` contains only `openai` and `pydantic` (no other packages)
  - Verify `openenv.yaml` parses successfully and contains `easy`, `medium`, `hard` task entries
  - _Requirements: 7.1, 7.2, 8.1, 8.2_

- [x] 7. Final checkpoint — Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Property tests use [Hypothesis](https://hypothesis.readthedocs.io/) with a minimum of 100 iterations each
- The `env/` module is reused as-is except for the `state()` accessor addition in task 1
- The React frontend, FastAPI backend, and Node/npm artifacts are not touched
