import os
import re
import json
from openai import OpenAI
from env import AIMEnv, Action, Grader
from env.models import TaskConfig


def get_env_config() -> tuple:
    """Read HF_TOKEN (required), API_BASE_URL, MODEL_NAME from environment."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    return (hf_token, api_base_url, model_name)


def format_start(task_name: str, env_name: str, model_name: str) -> str:
    return f"[START] task={task_name} env={env_name} model={model_name}"


def format_step(step: int, action: str, reward: float, done: bool, error) -> str:
    return (
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error is not None else 'null'}"
    )


def format_end(success: bool, steps: int, rewards: list) -> str:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    return f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}"


def run_episode(env: AIMEnv, client: OpenAI, model: str, task_name: str, env_name: str) -> None:
    obs = env.reset()
    print(format_start(task_name, env_name, model))

    done = False
    step_num = 0
    rewards_list = []

    while not done:
        error = None
        action = None

        # Build prompt
        inbox_lines = "\n".join(
            f"  - id={e.id} subject={e.subject!r} sender={e.sender} preview={e.preview!r}"
            for e in obs.inbox
        )
        prompt = (
            f"You are an email triage agent.\n\n"
            f"Inbox:\n{inbox_lines if inbox_lines else '  (empty)'}\n\n"
            f"Time left: {obs.time_left}  Step: {obs.step_count}\n\n"
            f"Available actions:\n"
            f'  {{"type": "open", "email_id": "<id>"}}\n'
            f'  {{"type": "classify", "email_id": "<id>", "category": "<urgent|normal|spam|promotions|social|updates|forums>", '
            f'"priority": "<low|medium|high|critical>", "route": "<inbox|archive|trash|escalate|review>"}}\n'
            f'  {{"type": "detect_phishing", "email_id": "<id>"}}\n'
            f'  {{"type": "submit"}}\n\n'
            f"Respond with a single JSON object representing your chosen action."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()
            # Extract JSON from possible markdown fences
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if m:
                raw = m.group(1)
            parsed = json.loads(raw)
            action = Action(**parsed)
        except Exception as exc:
            error = str(exc)
            action = Action(type="submit")

        action_str = action.type
        if action.email_id:
            action_str += f":{action.email_id}"

        obs, reward, done = env.step(action)
        rewards_list.append(reward.value)
        print(format_step(step_num, action_str, reward.value, done, error))
        step_num += 1

    result = env.get_result()
    score = Grader().grade_episode(result)
    print(format_end(score >= 0.5, step_num, rewards_list))


def main():
    hf_token, api_base_url, model_name = get_env_config()
    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    tasks = [
        ("easy", TaskConfig(seed=42, num_emails=3, time_budget=20,
                            ambiguity_level=0.0, has_phishing=False, time_pressure=0.0)),
        ("medium", TaskConfig(seed=137, num_emails=7, time_budget=30,
                              ambiguity_level=0.2, has_phishing=True, time_pressure=0.1)),
        ("hard", TaskConfig(seed=999, num_emails=12, time_budget=40,
                            ambiguity_level=0.5, has_phishing=True, time_pressure=0.5)),
    ]

    for task_name, config in tasks:
        env = AIMEnv(config)
        run_episode(env, client, model_name, task_name, "aim-email-triage")


if __name__ == "__main__":
    main()
