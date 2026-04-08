# Implementation Plan: inference-refactor

## Overview

Refactor `inference.py` in-place by introducing `EnvConfig`, `PromptBuilder`, `ActionParser`,
`EpisodeSummary`, `EpisodeRunner`, and `InferenceRunner` in dependency order. A companion test
file (`test_inference.py`) holds all unit and property-based tests. Public log format and task
definitions are preserved exactly.

## Tasks

- [x] 1. Define `EnvConfig` dataclass and `from_env` factory
  - Replace the `get_env_config()` function with a `@dataclass` class `EnvConfig` containing
    `hf_token`, `api_base_url`, `model_name`, and `timeout` fields with correct defaults.
  - Implement `from_env()` classmethod reading `HF_TOKEN` (required), `API_BASE_URL`,
    `MODEL_NAME`, and `INFERENCE_TIMEOUT` from `os.environ`.
  - Raise `ValueError("HF_TOKEN environment variable is required")` when `HF_TOKEN` is absent.
  - Add Google-style docstring and Python 3.10+ type hints.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 8.1, 8.2, 8.5_

  - [ ]* 1.1 Write unit tests for `EnvConfig.from_env`
    - Test missing `HF_TOKEN` → `ValueError` with exact message.
    - Test missing `API_BASE_URL` → default `"https://api.openai.com/v1"`.
    - Test missing `MODEL_NAME` → default `"gpt-4o-mini"`.
    - _Requirements: 1.2, 1.3, 1.4_

  - [ ]* 1.2 Write property test for `EnvConfig` field round-trip
    - **Property 1: EnvConfig field round-trip**
    - **Validates: Requirements 1.1**

- [x] 2. Implement `PromptBuilder`
  - Create stateless `PromptBuilder` class with a single `build(obs: Observation) -> str` method.
  - Render empty inbox as `"  (empty)"`.
  - Render each email as `  - id=<id> subject=<subject!r> sender=<sender> preview=<preview!r>`.
  - Include all four action schemas and `obs.time_left` / `obs.step_count`.
  - Add Google-style docstring and type hints.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 8.1, 8.2_

  - [ ]* 2.1 Write unit test for empty inbox rendering
    - Verify `"  (empty)"` appears in output when `obs.inbox == []`.
    - _Requirements: 2.2_

  - [ ]* 2.2 Write property test — prompt contains all action types
    - **Property 5: Prompt always contains all action types**
    - **Validates: Requirements 2.4**

  - [ ]* 2.3 Write property test — prompt reflects observation values
    - **Property 6: Prompt reflects observation values**
    - **Validates: Requirements 2.5**

  - [ ]* 2.4 Write property test — non-empty inbox renders all email fields
    - **Property 7: Non-empty inbox renders all email fields**
    - **Validates: Requirements 2.3**

- [x] 3. Implement `ActionParser`
  - Create `ActionParser` class with `parse(raw: str) -> Action` method.
  - Strip markdown fences via `re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)`.
  - Raise `ValueError` (wrapping original exception text) on `json.loads` failure.
  - Raise `ValueError` (wrapping original exception text) on `Action(**parsed)` failure.
  - Add Google-style docstring and type hints.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 8.1, 8.2_

  - [ ]* 3.1 Write property test — `ActionParser` round-trip
    - **Property 2: ActionParser round-trip**
    - **Validates: Requirements 3.3, 3.6**

  - [ ]* 3.2 Write property test — markdown-fenced JSON is transparent
    - **Property 3: Markdown-fenced JSON is transparent**
    - **Validates: Requirements 3.2**

  - [ ]* 3.3 Write property test — invalid JSON always raises `ValueError`
    - **Property 4: Invalid JSON always raises ValueError**
    - **Validates: Requirements 3.4**

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Define `EpisodeSummary` dataclass
  - Add `@dataclass EpisodeSummary` with fields `task_name: str`, `env_name: str`, `steps: int`,
    `rewards: list[float]`, and `success: bool`.
  - Add Google-style docstring.
  - _Requirements: 5.6, 8.2, 8.5_

- [x] 6. Implement `EpisodeRunner`
  - Create `EpisodeRunner.__init__(env, client, config)` storing all three as instance attributes
    and instantiating `PromptBuilder` and `ActionParser`.
  - Implement `run(task_name, env_name) -> EpisodeSummary`:
    - Call `env.reset()` exactly once.
    - Pass `timeout=config.timeout` to every `client.chat.completions.create(...)` call.
    - Catch `TimeoutError` / `ConnectionError` → `WARNING` log + `Action(type="submit")`.
    - Catch any other LLM exception → record error string + `Action(type="submit")`.
    - Catch `env.step` exception → `ERROR` log + `done = True`.
    - Emit `[STEP]` log lines at `INFO` with `error=<str|null>`.
    - Call `env.get_result()` and `Grader().grade_episode(result)` exactly once after loop.
    - Return `EpisodeSummary` with `success = score >= 0.5`.
  - Use list comprehension where applicable; add Google-style docstrings and type hints.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.3, 6.5, 8.1, 8.2, 8.4_

  - [ ]* 6.1 Write unit tests for `EpisodeRunner.run`
    - Test `env.reset()` called exactly once.
    - Test `env.get_result()` and `Grader().grade_episode()` called exactly once.
    - Test loop terminates when `env.step` raises.
    - _Requirements: 5.2, 5.4, 5.5_

  - [ ]* 6.2 Write property test — rewards length equals step count
    - **Property 8: EpisodeSummary rewards length equals step count**
    - **Validates: Requirements 5.3, 5.6**

  - [ ]* 6.3 Write property test — any LLM exception produces submit fallback
    - **Property 9: Any LLM exception produces submit fallback**
    - **Validates: Requirements 4.2, 4.4**

- [x] 7. Implement `InferenceRunner` and wire everything together
  - Define `DEFAULT_TASKS` class constant with the original three `TaskConfig` instances
    (easy/medium/hard, identical seeds/counts/budgets).
  - Implement `__init__(config=None, tasks=None)` — use `EnvConfig.from_env()` when `config` is
    `None`; use `DEFAULT_TASKS` when `tasks` is `None`.
  - Configure `"inference"` logger with a `StreamHandler` writing to `sys.stdout`.
  - Implement `run_all()`: emit `[START]` at `INFO`, create `EpisodeRunner`, call `run()`, emit
    `[END]` at `INFO`; catch unhandled exceptions → `ERROR` log + continue.
  - Replace the old `main()` function to instantiate and call `InferenceRunner().run_all()`.
  - Add Google-style docstrings and type hints throughout.
  - _Requirements: 6.1, 6.2, 6.4, 6.6, 7.1, 7.2, 7.3, 7.4, 8.1, 8.2_

  - [ ]* 7.1 Write unit tests for `InferenceRunner`
    - Test `DEFAULT_TASKS` used when no override provided.
    - Test `[START]` and `[END]` log lines match exact format strings.
    - _Requirements: 7.1, 6.2, 6.4_

  - [ ]* 7.2 Write property test — `run_all` executes every task
    - **Property 10: run_all executes every task**
    - **Validates: Requirements 7.2, 7.3**

- [x] 8. Apply code-style and static analysis fixes
  - Ensure all type annotations use Python 3.10+ union syntax (`X | None`).
  - Ensure all lines are ≤ 99 characters (PEP 8).
  - Replace any remaining `for`-loop appends with list comprehensions where applicable.
  - Verify `mypy --strict inference.py` passes with no errors.
  - Verify `ruff check inference.py` passes with no errors.
  - _Requirements: 8.1, 8.3, 8.4_

- [x] 9. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP.
- All tests live in a single `test_inference.py` file alongside `inference.py`.
- Property tests use Hypothesis (`@given`) with `settings(max_examples=25)`.
- Each property test carries a comment `# Feature: inference-refactor, Property <N>: <text>`.
- `inference.py` is the only production file modified; no new source files are created.
