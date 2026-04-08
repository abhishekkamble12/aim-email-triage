# Requirements Document

## Introduction

Refactor `inference.py` into a production-ready implementation for the OpenEnv RL Email Triage project.
The current file is a single-module script with monolithic functions, bare `except` clauses, no structured
logging, and no timeout handling. The refactor applies four pillars: Architecture & Modularity, Robustness
& Error Handling, Readability & Documentation, and Performance. The public behaviour (log format, task
definitions, grading threshold) must remain unchanged so existing CI pipelines are unaffected.

## Glossary

- **InferenceRunner**: The top-level orchestrator class responsible for iterating over tasks and running episodes.
- **EpisodeRunner**: The class responsible for executing a single RL episode loop (reset → step → grade).
- **PromptBuilder**: The class responsible for constructing the LLM prompt from an `Observation`.
- **ActionParser**: The class responsible for parsing raw LLM text into a validated `Action`.
- **EnvConfig**: A dataclass holding all runtime configuration values read from environment variables.
- **EpisodeSummary**: A dataclass capturing per-episode metrics (task name, steps, rewards, success flag).
- **AIMEnv**: The RL environment from the `env` package.
- **Action**: The Pydantic model from `env.models` representing an agent action.
- **Grader**: The evaluation utility from the `env` package that scores an `EpisodeResult`.
- **LLM**: Large Language Model accessed via the OpenAI-compatible client.

---

## Requirements

### Requirement 1: Configuration Management

**User Story:** As a platform engineer, I want all runtime configuration to be validated at startup, so that
misconfigured deployments fail fast with a clear error rather than crashing mid-episode.

#### Acceptance Criteria

1. THE `EnvConfig` SHALL be a `@dataclass` with fields `hf_token: str`, `api_base_url: str`, and `model_name: str`.
2. WHEN `HF_TOKEN` is absent from the environment, THE `EnvConfig` factory SHALL raise a `ValueError` with the message `"HF_TOKEN environment variable is required"`.
3. WHEN `API_BASE_URL` is absent from the environment, THE `EnvConfig` factory SHALL default to `"https://api.openai.com/v1"`.
4. WHEN `MODEL_NAME` is absent from the environment, THE `EnvConfig` factory SHALL default to `"gpt-4o-mini"`.
5. THE `EnvConfig` factory SHALL be a `@classmethod` named `from_env` that reads values from `os.environ`.

---

### Requirement 2: Prompt Construction

**User Story:** As an ML engineer, I want prompt construction isolated in a dedicated class, so that prompt
changes can be made and tested independently of the episode loop.

#### Acceptance Criteria

1. THE `PromptBuilder` SHALL expose a single public method `build(obs: Observation) -> str`.
2. WHEN `obs.inbox` is empty, THE `PromptBuilder` SHALL render the inbox section as `"  (empty)"`.
3. WHEN `obs.inbox` is non-empty, THE `PromptBuilder` SHALL render each email as `id=<id> subject=<subject> sender=<sender> preview=<preview>` using `!r` formatting for string fields.
4. THE `PromptBuilder` SHALL include all four action types (`open`, `classify`, `detect_phishing`, `submit`) with their full parameter schemas in the prompt.
5. THE `PromptBuilder` SHALL include `obs.time_left` and `obs.step_count` in the prompt.

---

### Requirement 3: Action Parsing

**User Story:** As an ML engineer, I want LLM output parsing isolated in a dedicated class, so that JSON
extraction and validation errors are handled in one place and do not propagate into the episode loop.

#### Acceptance Criteria

1. THE `ActionParser` SHALL expose a single public method `parse(raw: str) -> Action`.
2. WHEN the raw string contains a markdown-fenced JSON block, THE `ActionParser` SHALL extract the JSON from inside the fences before parsing.
3. WHEN `json.loads` succeeds and the resulting dict is a valid `Action`, THE `ActionParser` SHALL return that `Action`.
4. IF `json.loads` raises an exception, THEN THE `ActionParser` SHALL raise a `ValueError` with a message that includes the original exception text.
5. IF constructing `Action(**parsed)` raises an exception, THEN THE `ActionParser` SHALL raise a `ValueError` with a message that includes the original exception text.
6. FOR ALL valid JSON strings `s` that represent a valid `Action`, `ActionParser().parse(s)` SHALL return an `Action` equal to `Action(**json.loads(s))` (round-trip property).

---

### Requirement 4: LLM API Call Robustness

**User Story:** As a platform engineer, I want all LLM API calls to have timeout and retry handling, so that
transient network failures do not abort an entire evaluation run.

#### Acceptance Criteria

1. WHEN the LLM client call raises a `TimeoutError` or `ConnectionError`, THE `EpisodeRunner` SHALL catch the exception, log a warning, and fall back to `Action(type="submit")`.
2. WHEN the LLM client call raises any other exception, THE `EpisodeRunner` SHALL catch the exception, record the error string, and fall back to `Action(type="submit")`.
3. THE `EpisodeRunner` SHALL pass a `timeout` parameter (configurable via `EnvConfig`, defaulting to `30` seconds) to every LLM API call.
4. IF the `Action` fallback is used, THEN THE `EpisodeRunner` SHALL include the error string in the `[STEP]` log line's `error=` field.

---

### Requirement 5: Episode Execution

**User Story:** As an ML engineer, I want the episode loop encapsulated in `EpisodeRunner`, so that the
loop logic is testable and reusable across different task configurations.

#### Acceptance Criteria

1. THE `EpisodeRunner` SHALL accept `env: AIMEnv`, `client: OpenAI`, and `config: EnvConfig` in its constructor.
2. WHEN `EpisodeRunner.run(task_name: str, env_name: str) -> EpisodeSummary` is called, THE `EpisodeRunner` SHALL call `env.reset()` exactly once before the step loop.
3. WHILE `done` is `False`, THE `EpisodeRunner` SHALL call `env.step(action)` and append `reward.value` to the rewards list.
4. WHEN `env.step` raises an exception, THE `EpisodeRunner` SHALL log the error and terminate the loop by setting `done = True`.
5. WHEN the episode loop ends, THE `EpisodeRunner` SHALL call `env.get_result()` and `Grader().grade_episode(result)` exactly once.
6. THE `EpisodeRunner` SHALL return an `EpisodeSummary` dataclass with fields `task_name`, `env_name`, `steps`, `rewards`, and `success` (where `success` is `score >= 0.5`).

---

### Requirement 6: Structured Logging

**User Story:** As a platform engineer, I want all log output to use Python's `logging` module, so that log
level filtering and redirection work without modifying source code.

#### Acceptance Criteria

1. THE `InferenceRunner` SHALL configure a module-level `logging.Logger` named `"inference"`.
2. WHEN an episode starts, THE `InferenceRunner` SHALL emit a `[START]` log line at `INFO` level matching the format `[START] task=<name> env=<env> model=<model>`.
3. WHEN a step completes, THE `EpisodeRunner` SHALL emit a `[STEP]` log line at `INFO` level matching the format `[STEP] step=<n> action=<action> reward=<r:.2f> done=<done> error=<error|null>`.
4. WHEN an episode ends, THE `InferenceRunner` SHALL emit an `[END]` log line at `INFO` level matching the format `[END] success=<bool> steps=<n> rewards=<r1,r2,...>`.
5. WHEN an error is caught during an API call or env step, THE `EpisodeRunner` SHALL emit a log line at `WARNING` level before falling back.
6. THE `InferenceRunner` SHALL configure the root handler to write to `stdout` so existing CI log capture is unaffected.

---

### Requirement 7: Task Orchestration

**User Story:** As a platform engineer, I want task definitions and the main loop encapsulated in
`InferenceRunner`, so that adding or removing tasks requires no changes to the episode logic.

#### Acceptance Criteria

1. THE `InferenceRunner` SHALL define the three default tasks (`easy`, `medium`, `hard`) as a class-level constant using `TaskConfig` instances with the same seeds, email counts, and budgets as the original script.
2. WHEN `InferenceRunner.run_all()` is called, THE `InferenceRunner` SHALL iterate over all tasks and call `EpisodeRunner.run` for each one.
3. IF `EpisodeRunner.run` raises an unhandled exception for a task, THEN THE `InferenceRunner` SHALL log the error at `ERROR` level and continue to the next task.
4. THE `InferenceRunner` SHALL accept an optional `tasks` parameter in its constructor to allow overriding the default task list for testing.

---

### Requirement 8: Type Annotations and Code Style

**User Story:** As a staff engineer, I want all public interfaces to carry Python 3.10+ type annotations and
Google-style docstrings, so that IDEs and static analysis tools provide accurate feedback.

#### Acceptance Criteria

1. THE refactored module SHALL annotate all function and method signatures with Python 3.10+ type hints (using `X | None` union syntax instead of `Optional[X]`).
2. THE refactored module SHALL include Google-style docstrings on all public classes and methods.
3. THE refactored module SHALL comply with PEP 8 line-length limits (max 99 characters per line).
4. THE refactored module SHALL use list comprehensions in place of `for`-loop appends where the transformation is a single expression and clarity is not reduced.
5. THE refactored module SHALL use `dataclasses.dataclass` for `EnvConfig` and `EpisodeSummary` rather than plain tuples or dicts.
