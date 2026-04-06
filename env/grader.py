from .models import EpisodeResult
from typing import List

class Grader:
    def __init__(self):
        # Deterministic grader requires no API key
        pass

    def grade_episode(self, result: EpisodeResult, logs: List[str] = None) -> float:
        # Weighted formula as specified:
        # 0.30 * classification_acc
        # 0.20 * priority_acc
        # 0.20 * routing_acc
        # 0.20 * risk_score
        # 0.10 * efficiency_score
        
        final_score = (
            0.30 * result.classification_acc +
            0.20 * result.priority_acc +
            0.20 * result.routing_acc +
            0.20 * result.risk_score +
            0.10 * result.efficiency_score
        )
        
        # Ensure score is within bounds
        return min(1.0, max(0.0, final_score))