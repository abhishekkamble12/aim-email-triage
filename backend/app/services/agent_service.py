from app.schemas.request import TrainAgentRequest
from app.schemas.response import TrainingResponse, MetricsResponse
import random
import time

class AgentService:
    def train_agent(self, request: TrainAgentRequest) -> TrainingResponse:
        """Simulate RL training with realistic learning curve"""
        episodes = min(request.episodes, 1000)  # Cap at 1000 for safety
        learning_curve = []
        
        # Simulate training progress
        base_score = 0.1
        for i in range(episodes):
            # Learning curve: starts low, improves over time with some noise
            progress = i / episodes
            score = base_score + (0.8 * progress) + (random.random() * 0.1 - 0.05)
            score = min(0.95, max(0.05, score))  # Clamp between 0.05 and 0.95
            
            if i % max(1, episodes // 20) == 0:  # Sample every 5% of episodes
                learning_curve.append(round(score, 3))
        
        # Ensure we have at least 5 data points
        if len(learning_curve) < 5:
            learning_curve = [0.1, 0.25, 0.45, 0.65, 0.85][:len(learning_curve)]
        
        final_score = learning_curve[-1] if learning_curve else 0.85
        
        return TrainingResponse(
            status="completed",
            episodes_completed=episodes,
            final_score=final_score,
            metrics=MetricsResponse(
                accuracy=min(0.95, final_score * 0.9 + 0.1),
                efficiency=min(0.9, final_score * 0.8 + 0.2),
                risk_score=min(0.98, final_score * 0.95 + 0.05),
                learning_curve=learning_curve
            )
        )