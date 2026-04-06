"""
Hard Task Configuration Module.
"""
from env.models import TaskConfig

HARD_TASK_CONFIG = TaskConfig(
    num_emails=12,
    time_budget=40,
    seed=999,
    ambiguity_level=0.5,
    has_phishing=True,
    time_pressure=0.5
)
