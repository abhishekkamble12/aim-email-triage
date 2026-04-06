from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class EmailData(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    category: Optional[str] = None
    priority: Optional[str] = None
    route: Optional[str] = None
    is_phishing: Optional[bool] = None

class StepData(BaseModel):
    step: int
    action: str
    reward: float
    observation: Dict[str, Any]

class DemoResponse(BaseModel):
    emails: List[EmailData]
    steps: List[StepData]
    final_score: float
    agent_type: str

class MetricsResponse(BaseModel):
    accuracy: float
    efficiency: float
    risk_score: float
    learning_curve: List[float]

class TrainingResponse(BaseModel):
    status: str
    episodes_completed: int
    final_score: float
    metrics: MetricsResponse