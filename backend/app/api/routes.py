from fastapi import APIRouter, HTTPException
from ..services.env_service import EnvService
from ..services.training_service import TrainingService
from ..schemas.request import RunTaskRequest, TrainAgentRequest
from ..schemas.response import DemoResponse, TrainingResponse, MetricsResponse

router = APIRouter()
env_service = EnvService()
training_service = TrainingService()

@router.post("/run-demo", response_model=DemoResponse)
async def run_demo(request: RunTaskRequest):
    try:
        return env_service.run_task(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run-task", response_model=DemoResponse)
async def run_task(request: RunTaskRequest):
    try:
        return env_service.run_task(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-agent", response_model=TrainingResponse)
async def train_agent(request: TrainAgentRequest):
    try:
        return training_service.train_agent(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    try:
        metrics = env_service.get_metrics()
        return MetricsResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))