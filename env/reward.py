"""
Email RL Environment - Reward
Defines the Reward structure ensuring dictionary breakdown support.
"""

from pydantic import BaseModel, Field
from typing import Dict

class Reward(BaseModel):
    """
    Aggregated simulation reinforcement signal mapped to both absolute
    value float totals and a dictionary mapping of root causes.
    """
    value: float = Field(description="Calculated absolute float return.")
    components: Dict[str, float] = Field(description="Dictionary breaking down exact origins of the signal.")
