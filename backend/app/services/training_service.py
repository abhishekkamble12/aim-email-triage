from app.services.agent_service import AgentService
from app.schemas.request import TrainAgentRequest
from app.schemas.response import TrainingResponse

class TrainingService:
    def __init__(self):
        self.agent_service = AgentService()

    def train_agent(self, request: TrainAgentRequest) -> TrainingResponse:
        return self.agent_service.train_agent(request)