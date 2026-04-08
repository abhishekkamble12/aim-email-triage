"""Property-based tests for inference.py.

Feature: openenv-rl-execution
"""

import os

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from inference import EnvConfig, format_start


# ---------------------------------------------------------------------------
# Property 1: Missing HF_TOKEN raises ValueError
# Validates: Requirements 1.1
# ---------------------------------------------------------------------------


@given(st.none())
@settings(max_examples=20, deadline=None)
def test_missing_hf_token_raises_value_error(_):
    """**Validates: Requirements 1.1**

    For any execution environment where HF_TOKEN is not set,
    calling get_env_config() (EnvConfig.from_env()) SHALL raise a ValueError.
    """
    original = os.environ.pop("HF_TOKEN", None)
    try:
        with pytest.raises(ValueError):
            EnvConfig.from_env()
    finally:
        if original is not None:
            os.environ["HF_TOKEN"] = original


# ---------------------------------------------------------------------------
# Property 2: [START] line contains all required fields
# Validates: Requirements 2.1
# ---------------------------------------------------------------------------


@given(st.text(min_size=1), st.text(min_size=1), st.text(min_size=1))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_format_start_contains_required_fields(task_name, env_name, model_name):
    """**Validates: Requirements 2.1**

    For any combination of task name, env name, and model name,
    format_start() SHALL return a string containing [START], task=<value>,
    env=<value>, and model=<value>.
    """
    result = format_start(task_name, env_name, model_name)
    assert "[START]" in result
    assert f"task={task_name}" in result
    assert f"env={env_name}" in result
    assert f"model={model_name}" in result
