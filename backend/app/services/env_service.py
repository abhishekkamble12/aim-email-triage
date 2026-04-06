import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from env.env import AIMEnv
from tasks.task_easy import EASY_TASK_CONFIG
from tasks.task_medium import MEDIUM_TASK_CONFIG
from tasks.task_hard import HARD_TASK_CONFIG
from inference import HeuristicAgent, LLMAgent
from app.schemas.request import RunTaskRequest
from app.schemas.response import DemoResponse, EmailData, StepData
from app.core.config import settings
from typing import Dict, Any
import json

class EnvService:
    def __init__(self):
        self.env = None
        self.agent = None

    def _get_task_config(self, difficulty: str):
        configs = {
            "easy": EASY_TASK_CONFIG,
            "medium": MEDIUM_TASK_CONFIG,
            "hard": HARD_TASK_CONFIG
        }
        return configs.get(difficulty, EASY_TASK_CONFIG)

    def _get_agent(self, agent_type: str):
        if agent_type == "heuristic":
            return HeuristicAgent()
        elif agent_type == "llm":
            return LLMAgent()  # Will fallback to heuristic if no API key
        else:
            # For RL, we'd load a trained model, but for now use heuristic
            return HeuristicAgent()

    def run_task(self, request: RunTaskRequest) -> DemoResponse:
        task_config = self._get_task_config(request.difficulty)
        self.env = AIMEnv(task_config)
        self.agent = self._get_agent(request.agent_type)

        observation = self.env.reset()
        done = False
        steps = []
        step_count = 0

        while not done and step_count < settings.max_simulation_steps:  # Safety limit
            action = self.agent.decide(observation)
            next_observation, reward, done = self.env.step(action)

            step_data = StepData(
                step=step_count,
                action=self._serialize_action(action),
                reward=reward.value if hasattr(reward, 'value') else float(reward),
                observation=self._obs_to_dict(next_observation)
            )
            steps.append(step_data)

            observation = next_observation
            step_count += 1

        # Convert emails to EmailData
        emails = []
        for email in observation.inbox:
            emails.append(EmailData(
                id=email.id,
                subject=email.subject,
                sender=email.sender,
                body=email.preview,  # EmailPartial only has preview
                category=None,  # Don't leak ground truth
                priority=None,
                route=None,
                is_phishing=None
            ))

        final_score = self.env.get_score() if hasattr(self.env, 'get_score') else 0.5

        return DemoResponse(
            emails=emails,
            steps=steps,
            final_score=final_score,
            agent_type=request.agent_type
        )

    def _obs_to_dict(self, obs) -> Dict[str, Any]:
        """Convert Observation object to dict for frontend"""
        return {
            "inbox": [{"id": e.id, "subject": e.subject, "sender": e.sender, "body": e.preview} for e in obs.inbox],
            "opened": list(obs.opened),
            "time_left": obs.time_left,
            "step_count": obs.step_count,
            "pending_emails": obs.pending_emails,
            "alerts": obs.alerts,
            "classified": obs.classified,
            "prioritized": obs.prioritized,
            "routed": obs.routed
        }

    def _serialize_action(self, action) -> str:
        """Safely serialize action to JSON"""
        if hasattr(action, '__dict__'):
            return json.dumps(action.__dict__)
        elif isinstance(action, dict):
            return json.dumps(action)
        else:
            return str(action)

    def get_metrics(self) -> Dict[str, Any]:
        # Placeholder for metrics calculation
        return {
            "accuracy": 0.85,
            "efficiency": 0.72,
            "risk_score": 0.91,
            "learning_curve": [0.1, 0.3, 0.5, 0.7, 0.85]
        }