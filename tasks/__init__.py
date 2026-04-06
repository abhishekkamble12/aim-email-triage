"""
Task definitions providing configuration levels for the reinforcement agent.
"""

from .task_easy import EASY_TASK_CONFIG
from .task_medium import MEDIUM_TASK_CONFIG
from .task_hard import HARD_TASK_CONFIG

__all__ = [
    "EASY_TASK_CONFIG",
    "MEDIUM_TASK_CONFIG",
    "HARD_TASK_CONFIG"
]
