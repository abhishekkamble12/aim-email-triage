"""
Easy Task Configuration Module.
"""
from env.models import TaskConfig

EASY_TASK_CONFIG = TaskConfig(
    num_emails=3,
    time_budget=20,
    seed=42,
    ambiguity_level=0.0,
    has_phishing=False,
    time_pressure=0.0
)
