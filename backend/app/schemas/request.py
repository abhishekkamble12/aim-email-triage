from pydantic import BaseModel, validator
from typing import Optional

class RunTaskRequest(BaseModel):
    difficulty: str
    agent_type: Optional[str] = "heuristic"

    @validator('difficulty')
    def validate_difficulty(cls, v):
        if v not in ['easy', 'medium', 'hard']:
            raise ValueError('difficulty must be easy, medium, or hard')
        return v

    @validator('agent_type')
    def validate_agent_type(cls, v):
        if v not in ['heuristic', 'rl', 'llm']:
            raise ValueError('agent_type must be heuristic, rl, or llm')
        return v

class TrainAgentRequest(BaseModel):
    agent_type: str = "rl"
    episodes: int = 100

    @validator('agent_type')
    def validate_agent_type(cls, v):
        if v not in ['heuristic', 'rl', 'llm']:
            raise ValueError('agent_type must be heuristic, rl, or llm')
        return v

    @validator('episodes')
    def validate_episodes(cls, v):
        if not (1 <= v <= 1000):
            raise ValueError('episodes must be between 1 and 1000')
        return v