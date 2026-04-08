# Requirements Document

## Introduction

Refactor `inference.py` to expose the OpenEnv HTTP API so the AIM-Env can be driven externally by the Round 1 automated grader. The server must remove the internal LLM agent loop, instantiate `AIMEnv` as a global singleton, and expose `/reset`, `/step`, and `/state` endpoints that accept and return JSON-serializable Pydantic models. The server must bind to `0.0.0.0:7860` so the Hugging Face Space remains green.

## Glossary

- **Inference_Script**: The single `inference.py` file in the repository root that serves as the sole entry point for the RL execution environment.
- **OpenEnv_API**: The standardized HTTP interface (`/reset`, `/step`, `/state`) consumed by the automated grader to drive an RL episode.
- **AIMEnv**: The `AIMEnv` class from `env/env.py` that implements the RL environment logic.
- **Env_Singleton**: The single global instance of `AIMEnv` that persists across HTTP requests for the lifetime of the server process.
- **Observation**: The Pydantic model returned by `AIMEnv.reset()`, `AIMEnv.step()`, and `AIMEnv.state()` representing the current environment state.
- **Action**: The Pydantic model accepted by `AIMEnv.step()` representing the agent's chosen operation.
- **Reward**: The Pydantic model returned by `AIMEnv.step()` with a `.value` float field representing the scalar reinforcement signal.
- **Grader**: The external automated scoring process that acts as the agent, calling `/reset`, `/step`, and `/state` over HTTP.
- **TaskConfig**: The Pydantic model used to configure an `AIMEnv` instance with seed, email count, time budget, and difficulty parameters.

## Requirements

### Requirement 1: Global Environment Singleton

**User Story:** As the Grader, I want the environment to persist between API calls, so that episode state is maintained across `/reset`, `/step`, and `/state` requests within a single server process.

#### Acceptance Criteria

1. THE Inference_Script SHALL instantiate a single Env_Singleton of `AIMEnv` at module load time using a default `TaskConfig`.
2. THE Inference_Script SHALL reuse the same Env_Singleton instance for all `/reset`, `/step`, and `/state` requests without re-instantiating between calls.
3. WHEN the server process starts, THE Inference_Script SHALL initialize the Env_Singleton before the FastAPI application begins accepting requests.

---

### Requirement 2: POST /reset Endpoint

**User Story:** As the Grader, I want to reset the environment to an initial state, so that I can start a new episode with a known or random seed.

#### Acceptance Criteria

1. THE Inference_Script SHALL expose a `POST /reset` route on the FastAPI application.
2. WHEN a `POST /reset` request is received with a JSON body containing a `seed` integer field, THE Inference_Script SHALL call `env.reset()` on the Env_Singleton (passing the seed to `TaskConfig` or equivalent initialization).
3. WHEN a `POST /reset` request is received with no body or an empty body, THE Inference_Script SHALL call `env.reset()` using the existing `TaskConfig` seed.
4. WHEN `env.reset()` returns an Observation, THE Inference_Script SHALL return a `200 OK` JSON response containing the Observation serialized via `.model_dump()`.
5. IF `env.reset()` raises an exception, THEN THE Inference_Script SHALL return a `500` JSON response with an `{"error": "<message>"}` body.

---

### Requirement 3: POST /step Endpoint

**User Story:** As the Grader, I want to advance the environment by one step, so that I can drive the episode forward with my chosen action.

#### Acceptance Criteria

1. THE Inference_Script SHALL expose a `POST /step` route on the FastAPI application.
2. WHEN a `POST /step` request is received with a JSON body containing an `action` field, THE Inference_Script SHALL construct an `Action` model from the `action` value and call `env.step(action)` on the Env_Singleton.
3. WHEN `env.step()` returns `(observation, reward, done)`, THE Inference_Script SHALL return a `200 OK` JSON response with the exact structure: `{"observation": <Observation.model_dump()>, "reward": <float rounded to 2 decimal places>, "done": <bool>, "info": {}}`.
4. THE Inference_Script SHALL round the reward to 2 decimal places using `round(reward.value, 2)` before including it in the response.
5. IF the `action` field is missing from the request body, THEN THE Inference_Script SHALL return a `422` response.
6. IF `env.step()` raises a `RuntimeError` (episode already ended), THEN THE Inference_Script SHALL return a `400` JSON response with an `{"error": "<message>"}` body.
7. IF `env.step()` raises any other exception, THEN THE Inference_Script SHALL return a `500` JSON response with an `{"error": "<message>"}` body.

---

### Requirement 4: GET /state Endpoint

**User Story:** As the Grader, I want to read the current environment state without advancing the episode, so that I can inspect the observation at any point.

#### Acceptance Criteria

1. THE Inference_Script SHALL expose a `GET /state` route on the FastAPI application.
2. WHEN a `GET /state` request is received, THE Inference_Script SHALL call `env.state()` on the Env_Singleton and return a `200 OK` JSON response containing the Observation serialized via `.model_dump()`.
3. IF `env.state()` raises a `RuntimeError` (reset not yet called), THEN THE Inference_Script SHALL return a `400` JSON response with an `{"error": "<message>"}` body.

---

### Requirement 5: JSON Serialization

**User Story:** As the Grader, I want all API responses to be valid JSON, so that my HTTP client can parse them without errors.

#### Acceptance Criteria

1. THE Inference_Script SHALL serialize all Pydantic model responses by calling `.model_dump()` before returning them in `JSONResponse` or as FastAPI response bodies.
2. THE Inference_Script SHALL ensure no `TypeError: Object of type X is not JSON serializable` is raised for any `/reset`, `/step`, or `/state` response.
3. WHERE a response field contains a nested Pydantic model, THE Inference_Script SHALL recursively serialize it so the final response contains only JSON-native types (dict, list, str, int, float, bool, None).

---

### Requirement 6: Remove Agent Threading

**User Story:** As the Grader, I want the server to respond synchronously to HTTP requests, so that the Grader can act as the sole agent without interference from an internal LLM loop.

#### Acceptance Criteria

1. THE Inference_Script SHALL NOT start the `_inference_worker` background thread on application startup.
2. THE Inference_Script SHALL remove or disable the `_lifespan` context manager that launches the background thread.
3. THE Inference_Script SHALL retain the `/`, `/health`, and `/status` routes for operational monitoring.
4. WHEN the server starts, THE Inference_Script SHALL set the run status to `"ready"` instead of `"pending"` to indicate it is waiting for external Grader control.

---

### Requirement 7: Server Binding

**User Story:** As the Hugging Face Space operator, I want the server to bind to the correct host and port, so that the Space status remains green and the Grader can reach the API.

#### Acceptance Criteria

1. THE Inference_Script SHALL bind the uvicorn server to host `0.0.0.0` and port `7860`.
2. WHEN the `main()` function is called, THE Inference_Script SHALL start uvicorn with `host="0.0.0.0"` and `port=7860`.
