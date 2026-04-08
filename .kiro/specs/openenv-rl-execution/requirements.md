# Requirements Document

## Introduction

Refactor the AIM-Env Platform from a full-stack web application (React frontend + FastAPI backend) into a strictly compliant, lightweight, single-script Reinforcement Learning execution environment. The result must be a self-contained `inference.py` that passes the Round 1 automated grader with a perfect score, backed by a lean Docker image and an `openenv.yaml` config.

## Glossary

- **Inference_Script**: The single `inference.py` file in the repository root that serves as the sole entry point for the RL execution environment.
- **OpenEnv**: The standardized RL environment specification that defines typed models, API functions, and stdout formatting rules used by the automated grader.
- **LLM_Client**: The `openai.OpenAI` Python client instance used to communicate with the language model.
- **Grader**: The deterministic scoring function that evaluates an episode result and returns a float score in the range [0.0, 1.0].
- **Task**: A single RL episode configuration (Easy, Medium, or Hard) with a fixed seed, email count, time budget, and difficulty parameters.
- **Observation**: A typed Pydantic model representing the agent view of the environment state at a given step.
- **Action**: A typed Pydantic model representing the agent chosen operation at a given step.
- **Reward**: A typed Pydantic model representing the scalar reinforcement signal and its component breakdown for a step.
- **Episode**: A complete run of one Task from reset to terminal state.
- **Docker_Image**: The container image built from the root Dockerfile used to execute the Inference_Script in the grader environment.

