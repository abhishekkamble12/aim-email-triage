"""Tests for inference.py — unit and property tests for all classes."""

import json
import logging
import os
from unittest.mock import MagicMock, patch, call
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from env.models import Action, EmailPartial, Observation, TaskConfig
from inference import (
    ActionParser,
    EnvConfig,
    EpisodeRunner,
    EpisodeSummary,
    InferenceRunner,
    PromptBuilder,
)


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

_ACTION_TYPES = ["open", "classify", "detect_phishing", "submit"]
_CATEGORIES = ["urgent", "normal", "spam", "promotions", "social", "updates", "forums"]
_PRIORITIES = ["low", "medium", "high", "critical"]
_ROUTES = ["inbox", "archive", "trash", "escalate", "review"]


def _make_observation(
    inbox: list[EmailPartial] | None = None,
    time_left: int = 10,
    step_count: int = 0,
) -> Observation:
    return Observation(
        inbox=inbox or [],
        opened=[],
        time_left=time_left,
        step_count=step_count,
        pending_emails=0,
        alerts=[],
        classified=0,
        prioritized=0,
        routed=0,
    )


def _make_email(
    id: str = "e1",
    subject: str = "Hello",
    sender: str = "a@b.com",
    preview: str = "Hi there",
) -> EmailPartial:
    return EmailPartial(id=id, subject=subject, sender=sender, preview=preview)


def _make_env_config() -> EnvConfig:
    return EnvConfig(
        hf_token="tok",
        api_base_url="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        timeout=30,
    )


# Hypothesis strategy for valid Action instances
@st.composite
def action_strategy(draw: st.DrawFn) -> Action:
    action_type = draw(st.sampled_from(_ACTION_TYPES))
    email_id = draw(st.one_of(
        st.none(),
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=20),
    ))
    if action_type == "classify":
        category = draw(st.sampled_from(_CATEGORIES))
        priority = draw(st.sampled_from(_PRIORITIES))
        route = draw(st.sampled_from(_ROUTES))
        return Action(
            type=action_type,
            email_id=email_id,
            category=category,
            priority=priority,
            route=route,
        )
    return Action(type=action_type, email_id=email_id)


# Hypothesis strategy for Observation instances
@st.composite
def observation_strategy(draw: st.DrawFn) -> Observation:
    time_left = draw(st.integers(min_value=0, max_value=1000))
    step_count = draw(st.integers(min_value=0, max_value=1000))
    emails = draw(
        st.lists(
            st.builds(
                EmailPartial,
                id=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=10),
                subject=_ascii_any,
                sender=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789@.", min_size=1, max_size=30),
                preview=_ascii_any,
            ),
            min_size=0,
            max_size=5,
        )
    )
    return Observation(
        inbox=emails,
        opened=[],
        time_left=time_left,
        step_count=step_count,
        pending_emails=0,
        alerts=[],
        classified=0,
        prioritized=0,
        routed=0,
    )


# ---------------------------------------------------------------------------
# Task 1.1 — Unit tests for EnvConfig.from_env
# ---------------------------------------------------------------------------


def test_from_env_missing_hf_token_raises(monkeypatch):
    """Missing HF_TOKEN must raise ValueError with the exact required message."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HF_TOKEN environment variable is required"):
        EnvConfig.from_env()


def test_from_env_missing_api_base_url_uses_default(monkeypatch):
    """Absent API_BASE_URL must default to 'https://api.openai.com/v1'."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.delenv("API_BASE_URL", raising=False)
    config = EnvConfig.from_env()
    assert config.api_base_url == "https://api.openai.com/v1"


def test_from_env_missing_model_name_uses_default(monkeypatch):
    """Absent MODEL_NAME must default to 'gpt-4o-mini'."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.delenv("MODEL_NAME", raising=False)
    config = EnvConfig.from_env()
    assert config.model_name == "gpt-4o-mini"


def test_from_env_missing_timeout_uses_default(monkeypatch):
    """Absent INFERENCE_TIMEOUT must default to 30."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.delenv("INFERENCE_TIMEOUT", raising=False)
    config = EnvConfig.from_env()
    assert config.timeout == 30


def test_from_env_reads_all_values(monkeypatch):
    """All env vars present must be reflected in the returned EnvConfig."""
    monkeypatch.setenv("HF_TOKEN", "my-token")
    monkeypatch.setenv("API_BASE_URL", "https://custom.example.com/v1")
    monkeypatch.setenv("MODEL_NAME", "llama-3")
    monkeypatch.setenv("INFERENCE_TIMEOUT", "60")
    config = EnvConfig.from_env()
    assert config.hf_token == "my-token"
    assert config.api_base_url == "https://custom.example.com/v1"
    assert config.model_name == "llama-3"
    assert config.timeout == 60


# ---------------------------------------------------------------------------
# Task 1.2 — Property test: EnvConfig field round-trip
# Feature: inference-refactor, Property 1: EnvConfig field round-trip
# Validates: Requirements 1.1
# ---------------------------------------------------------------------------


_ascii = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.",
    min_size=1,
    max_size=64,
)
_ascii_any = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.",
    min_size=0,
    max_size=50,
)


@given(
    hf_token=_ascii,
    api_base_url=_ascii,
    model_name=_ascii,
    timeout=st.integers(min_value=1, max_value=3600),
)
@settings(max_examples=25, deadline=None)
def test_envconfig_field_round_trip(hf_token, api_base_url, model_name, timeout):
    # Feature: inference-refactor, Property 1: EnvConfig field round-trip
    # Validates: Requirements 1.1
    config = EnvConfig(
        hf_token=hf_token,
        api_base_url=api_base_url,
        model_name=model_name,
        timeout=timeout,
    )
    assert config.hf_token == hf_token
    assert config.api_base_url == api_base_url
    assert config.model_name == model_name
    assert config.timeout == timeout


# ---------------------------------------------------------------------------
# Task 2.1 — Unit test: empty inbox rendering
# ---------------------------------------------------------------------------


def test_prompt_builder_empty_inbox():
    """PromptBuilder must render '  (empty)' when obs.inbox is empty."""
    obs = _make_observation(inbox=[])
    prompt = PromptBuilder().build(obs)
    assert "  (empty)" in prompt


def test_prompt_builder_non_empty_inbox():
    """PromptBuilder must render email fields when inbox is non-empty."""
    email = _make_email(id="e1", subject="Test", sender="x@y.com", preview="Hello")
    obs = _make_observation(inbox=[email])
    prompt = PromptBuilder().build(obs)
    assert "id=e1" in prompt
    assert "subject='Test'" in prompt
    assert "sender=x@y.com" in prompt
    assert "preview='Hello'" in prompt


# ---------------------------------------------------------------------------
# Task 2.2 — Property test: prompt contains all action types
# Feature: inference-refactor, Property 5: Prompt always contains all action types
# Validates: Requirements 2.4
# ---------------------------------------------------------------------------


@given(obs=observation_strategy())
@settings(max_examples=25, deadline=None)
def test_prompt_contains_all_action_types(obs):
    # Feature: inference-refactor, Property 5: Prompt always contains all action types
    # Validates: Requirements 2.4
    prompt = PromptBuilder().build(obs)
    for action_type in ("open", "classify", "detect_phishing", "submit"):
        assert action_type in prompt


# ---------------------------------------------------------------------------
# Task 2.3 — Property test: prompt reflects observation values
# Feature: inference-refactor, Property 6: Prompt reflects observation values
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------


@given(obs=observation_strategy())
@settings(max_examples=25, deadline=None)
def test_prompt_reflects_obs_values(obs):
    # Feature: inference-refactor, Property 6: Prompt reflects observation values
    # Validates: Requirements 2.5
    prompt = PromptBuilder().build(obs)
    assert str(obs.time_left) in prompt
    assert str(obs.step_count) in prompt


# ---------------------------------------------------------------------------
# Task 2.4 — Property test: non-empty inbox renders all email fields
# Feature: inference-refactor, Property 7: Non-empty inbox renders all email fields
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------


@given(
    obs=observation_strategy().filter(lambda o: len(o.inbox) > 0)
)
@settings(max_examples=25, deadline=None)
def test_non_empty_inbox_renders_all_fields(obs):
    # Feature: inference-refactor, Property 7: Non-empty inbox renders all email fields
    # Validates: Requirements 2.3
    prompt = PromptBuilder().build(obs)
    for email in obs.inbox:
        assert email.id in prompt
        assert repr(email.subject) in prompt
        assert email.sender in prompt
        assert repr(email.preview) in prompt


# ---------------------------------------------------------------------------
# Task 3.1 — Property test: ActionParser round-trip
# Feature: inference-refactor, Property 2: ActionParser round-trip
# Validates: Requirements 3.3, 3.6
# ---------------------------------------------------------------------------


@given(action=action_strategy())
@settings(max_examples=25, deadline=None)
def test_action_parser_round_trip(action):
    # Feature: inference-refactor, Property 2: ActionParser round-trip
    # Validates: Requirements 3.3, 3.6
    raw = json.dumps(action.model_dump())
    parsed = ActionParser().parse(raw)
    assert parsed == action


# ---------------------------------------------------------------------------
# Task 3.2 — Property test: markdown-fenced JSON is transparent
# Feature: inference-refactor, Property 3: Markdown-fenced JSON is transparent
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------


@given(action=action_strategy())
@settings(max_examples=25, deadline=None)
def test_markdown_fenced_json_transparent(action):
    # Feature: inference-refactor, Property 3: Markdown-fenced JSON is transparent
    # Validates: Requirements 3.2
    raw = json.dumps(action.model_dump())
    fenced = f"```json\n{raw}\n```"
    assert ActionParser().parse(fenced) == ActionParser().parse(raw)


# ---------------------------------------------------------------------------
# Task 3.3 — Property test: invalid JSON always raises ValueError
# Feature: inference-refactor, Property 4: Invalid JSON always raises ValueError
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------


def _is_invalid_json(s: str) -> bool:
    try:
        json.loads(s)
        return False
    except Exception:
        return True


@given(text=st.one_of(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=30),  # plain words, never valid JSON objects
    st.just("not json"),
    st.just("{ bad json }"),
    st.just(""),
    st.just("null"),  # valid JSON but not a dict → Action(**None) fails
).filter(_is_invalid_json))
@settings(max_examples=25, deadline=None)
def test_invalid_json_raises_value_error(text):
    # Feature: inference-refactor, Property 4: Invalid JSON always raises ValueError
    # Validates: Requirements 3.4
    with pytest.raises(ValueError):
        ActionParser().parse(text)


# ---------------------------------------------------------------------------
# Task 6.1 — Unit tests for EpisodeRunner.run
# ---------------------------------------------------------------------------


def _make_mock_env(num_steps: int = 2, score: float = 1.0):
    """Build a mock AIMEnv that runs for num_steps then sets done=True."""
    env = MagicMock()
    obs = _make_observation()
    env.reset.return_value = obs

    reward = MagicMock()
    reward.value = 1.0

    # Each step returns (obs, reward, done); done=True on last step
    step_returns = [(obs, reward, False)] * (num_steps - 1) + [(obs, reward, True)]
    env.step.side_effect = step_returns

    result = MagicMock()
    env.get_result.return_value = result

    grader_instance = MagicMock()
    grader_instance.grade_episode.return_value = score

    return env, grader_instance, result


def _make_mock_client(response_text: str = '{"type": "submit"}'):
    client = MagicMock()
    msg = MagicMock()
    msg.content = response_text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    return client


def test_episode_runner_reset_called_once():
    """env.reset() must be called exactly once."""
    env, grader_mock, result = _make_mock_env(num_steps=1)
    client = _make_mock_client()
    config = _make_env_config()

    with patch("inference.Grader", return_value=grader_mock):
        runner = EpisodeRunner(env, client, config)
        runner.run("easy", "aim-email-triage")

    env.reset.assert_called_once()


def test_episode_runner_get_result_and_grade_called_once():
    """env.get_result() and Grader().grade_episode() must each be called once."""
    env, grader_mock, result = _make_mock_env(num_steps=1)
    client = _make_mock_client()
    config = _make_env_config()

    with patch("inference.Grader", return_value=grader_mock):
        runner = EpisodeRunner(env, client, config)
        runner.run("easy", "aim-email-triage")

    env.get_result.assert_called_once()
    grader_mock.grade_episode.assert_called_once_with(result)


def test_episode_runner_terminates_on_env_step_exception():
    """Loop must terminate when env.step raises an exception."""
    env = MagicMock()
    obs = _make_observation()
    env.reset.return_value = obs
    env.step.side_effect = RuntimeError("env exploded")
    result = MagicMock()
    env.get_result.return_value = result

    grader_mock = MagicMock()
    grader_mock.grade_episode.return_value = 0.0

    client = _make_mock_client()
    config = _make_env_config()

    with patch("inference.Grader", return_value=grader_mock):
        runner = EpisodeRunner(env, client, config)
        summary = runner.run("easy", "aim-email-triage")

    # env.step raised, so loop terminated; steps should be 0
    assert summary.steps == 0


# ---------------------------------------------------------------------------
# Task 6.2 — Property test: rewards length equals step count
# Feature: inference-refactor, Property 8: EpisodeSummary rewards length equals step count
# Validates: Requirements 5.3, 5.6
# ---------------------------------------------------------------------------


@given(num_steps=st.integers(min_value=1, max_value=10))
@settings(max_examples=25, deadline=None)
def test_rewards_length_equals_step_count(num_steps):
    # Feature: inference-refactor, Property 8: EpisodeSummary rewards length equals step count
    # Validates: Requirements 5.3, 5.6
    env = MagicMock()
    obs = _make_observation()
    env.reset.return_value = obs

    reward = MagicMock()
    reward.value = 1.0
    step_returns = [(obs, reward, False)] * (num_steps - 1) + [(obs, reward, True)]
    env.step.side_effect = step_returns

    result = MagicMock()
    env.get_result.return_value = result

    grader_mock = MagicMock()
    grader_mock.grade_episode.return_value = 1.0

    client = _make_mock_client()
    config = _make_env_config()

    with patch("inference.Grader", return_value=grader_mock):
        runner = EpisodeRunner(env, client, config)
        summary = runner.run("easy", "aim-email-triage")

    assert len(summary.rewards) == num_steps
    assert summary.steps == num_steps


# ---------------------------------------------------------------------------
# Task 6.3 — Property test: any LLM exception produces submit fallback
# Feature: inference-refactor, Property 9: Any LLM exception produces submit fallback
# Validates: Requirements 4.2, 4.4
# ---------------------------------------------------------------------------

_EXCEPTION_TYPES = [
    ValueError("bad value"),
    RuntimeError("runtime error"),
    TimeoutError("timed out"),
    ConnectionError("connection refused"),
    OSError("os error"),
    Exception("generic error"),
]


@given(exc=st.sampled_from(_EXCEPTION_TYPES))
@settings(max_examples=25, deadline=None)
def test_llm_exception_produces_submit_fallback(exc):
    # Feature: inference-refactor, Property 9: Any LLM exception produces submit fallback
    # Validates: Requirements 4.2, 4.4
    env = MagicMock()
    obs = _make_observation()
    env.reset.return_value = obs

    reward = MagicMock()
    reward.value = 0.0
    # One step: LLM raises, fallback submit action is used, done=True
    env.step.return_value = (obs, reward, True)

    result = MagicMock()
    env.get_result.return_value = result

    grader_mock = MagicMock()
    grader_mock.grade_episode.return_value = 0.0

    client = MagicMock()
    client.chat.completions.create.side_effect = exc

    config = _make_env_config()

    with patch("inference.Grader", return_value=grader_mock):
        runner = EpisodeRunner(env, client, config)
        summary = runner.run("easy", "aim-email-triage")

    # The action passed to env.step must be a submit action
    called_action = env.step.call_args[0][0]
    assert called_action.type == "submit"


# ---------------------------------------------------------------------------
# Task 7.1 — Unit tests for InferenceRunner
# ---------------------------------------------------------------------------


def test_inference_runner_uses_default_tasks(monkeypatch):
    """InferenceRunner must use DEFAULT_TASKS when no tasks override is given."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    config = _make_env_config()
    runner = InferenceRunner(config=config)
    assert runner.tasks is InferenceRunner.DEFAULT_TASKS


def test_inference_runner_uses_provided_tasks(monkeypatch):
    """InferenceRunner must use the provided tasks list when given."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    config = _make_env_config()
    custom_tasks = [
        ("custom", TaskConfig(seed=1, num_emails=1, time_budget=5)),
    ]
    runner = InferenceRunner(config=config, tasks=custom_tasks)
    assert runner.tasks is custom_tasks


def test_inference_runner_start_log_format(monkeypatch, caplog):
    """[START] log line must match the exact required format."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    config = _make_env_config()

    mock_summary = EpisodeSummary(
        task_name="easy",
        env_name="aim-email-triage",
        steps=1,
        rewards=[1.0],
        success=True,
    )

    tasks = [("easy", TaskConfig(seed=42, num_emails=1, time_budget=5))]

    with patch("inference.AIMEnv"), patch("inference.EpisodeRunner") as MockRunner:
        MockRunner.return_value.run.return_value = mock_summary
        with patch("inference.OpenAI"):
            runner = InferenceRunner(config=config, tasks=tasks)
            with caplog.at_level(logging.INFO, logger="inference"):
                runner.run_all()

    start_lines = [r for r in caplog.records if "[START]" in r.message]
    assert len(start_lines) == 1
    assert "task=easy" in start_lines[0].message
    assert "env=aim-email-triage" in start_lines[0].message
    assert f"model={config.model_name}" in start_lines[0].message


def test_inference_runner_end_log_format(monkeypatch, caplog):
    """[END] log line must match the exact required format."""
    monkeypatch.setenv("HF_TOKEN", "tok")
    config = _make_env_config()

    mock_summary = EpisodeSummary(
        task_name="easy",
        env_name="aim-email-triage",
        steps=2,
        rewards=[0.5, 1.0],
        success=True,
    )

    tasks = [("easy", TaskConfig(seed=42, num_emails=1, time_budget=5))]

    with patch("inference.AIMEnv"), patch("inference.EpisodeRunner") as MockRunner:
        MockRunner.return_value.run.return_value = mock_summary
        with patch("inference.OpenAI"):
            runner = InferenceRunner(config=config, tasks=tasks)
            with caplog.at_level(logging.INFO, logger="inference"):
                runner.run_all()

    end_lines = [r for r in caplog.records if "[END]" in r.message]
    assert len(end_lines) == 1
    assert "success=true" in end_lines[0].message
    assert "steps=2" in end_lines[0].message
    assert "0.50,1.00" in end_lines[0].message


# ---------------------------------------------------------------------------
# Task 7.2 — Property test: run_all executes every task
# Feature: inference-refactor, Property 10: run_all executes every task
# Validates: Requirements 7.2, 7.3
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=25, deadline=None)
def test_run_all_executes_every_task(n):
    # Feature: inference-refactor, Property 10: run_all executes every task
    # Validates: Requirements 7.2, 7.3
    config = _make_env_config()

    tasks = [
        (f"task_{i}", TaskConfig(seed=i, num_emails=1, time_budget=5))
        for i in range(n)
    ]

    mock_summary = EpisodeSummary(
        task_name="t",
        env_name="aim-email-triage",
        steps=0,
        rewards=[],
        success=False,
    )

    with patch("inference.AIMEnv"), patch("inference.EpisodeRunner") as MockRunner:
        MockRunner.return_value.run.return_value = mock_summary
        with patch("inference.OpenAI"):
            runner = InferenceRunner(config=config, tasks=tasks)
            runner.run_all()

    assert MockRunner.return_value.run.call_count == n
