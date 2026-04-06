"""
Medium Task Configuration Module.
"""
from env.models import TaskConfig

MEDIUM_TASK_CONFIG = TaskConfig(
    num_emails=7,
    time_budget=30,
    seed=137,
    ambiguity_level=0.2,
    has_phishing=True,
    time_pressure=0.1
)
