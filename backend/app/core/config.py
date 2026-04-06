import os
from pydantic import BaseModel
from typing import List

class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    app_name: str = "AIM-Env Platform"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    allowed_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    max_simulation_steps: int = int(os.getenv("MAX_SIMULATION_STEPS", "100"))
    simulation_timeout: int = int(os.getenv("SIMULATION_TIMEOUT", "30"))

settings = Settings()