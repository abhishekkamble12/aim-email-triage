"""
Email RL Environment - Models
Defines all Pydantic schemas and enum structures representing the state
and configuration of the simulation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum

class EmailCategory(str, Enum):
    """Categories assigned to emails for processing classification."""
    urgent = "urgent"
    normal = "normal"
    spam = "spam"
    promotions = "promotions"
    social = "social"
    updates = "updates"
    forums = "forums"

class PriorityLevel(str, Enum):
    """Triage priority levels representing urgency or importance."""
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class RouteOption(str, Enum):
    """End-state destinations for parsed emails."""
    inbox = "inbox"
    archive = "archive"
    trash = "trash"
    escalate = "escalate"
    review = "review"

class Action(BaseModel):
    """
    Action space provided by the generic Agent interacting with the environment.
    Types can be: 'open', 'classify', 'detect_phishing', 'submit'
    """
    type: str = Field(description="The generic operation type to perform.")
    email_id: Optional[str] = Field(default=None, description="The targeted email unique identifier.")
    category: Optional[EmailCategory] = Field(default=None, description="The category classification assigned.")
    priority: Optional[PriorityLevel] = Field(default=None, description="The priority classification assigned.")
    route: Optional[RouteOption] = Field(default=None, description="The destination route assigned.")

    @field_validator("email_id", mode="before")
    @classmethod
    def coerce_email_id_to_str(cls, v: object) -> object:
        """Coerce integer email_id to string.

        The LLM sometimes emits {"email_id": 28289} (no quotes) instead of
        {"email_id": "28289"}, which causes a Pydantic validation error.
        Converting to str here silently fixes that without touching env logic.
        """
        if v is not None and not isinstance(v, str):
            return str(v)
        return v

class EmailPartial(BaseModel):
    """Structure for emails visible from the inbox viewport."""
    id: str = Field(description="Unique email identifier.")
    subject: str = Field(description="Email subject line.")
    sender: str = Field(description="Original sender email address.")
    preview: str = Field(description="Initial text preview excerpt (max 50 chars).")

class EmailFull(EmailPartial):
    """Fully parsed email content visible only when explicitly 'opened'."""
    body: str = Field(description="The complete email body content.")

class EmailGroundTruth(BaseModel):
    """
    Simulated reality. Absolute ground truth metrics governing the true
    attributes of generated emails. Absolutely hidden from the agent.
    """
    id: str = Field(description="Target mapped email identifier.")
    is_phishing: bool = Field(description="Whether the email contains malicious intent.")
    true_category: EmailCategory = Field(description="The true intrinsic category.")
    true_priority: PriorityLevel = Field(description="The intrinsic priority requirements.")
    true_route: RouteOption = Field(description="The optimal destination for correct triage.")

class Observation(BaseModel):
    """
    The observation state vector provided back to the agent detailing
    everything currently visible and known at the current step threshold.
    """
    inbox: List[EmailPartial] = Field(description="Emails currently visible in the inbox.")
    opened: List[str] = Field(description="List of currently opened email IDs.")
    time_left: int = Field(description="Remaining time units before deadline failure.")
    step_count: int = Field(description="Current internal step tracker count.")
    pending_emails: int = Field(description="Total count of emails remaining unread/unprocessed.")
    alerts: List[str] = Field(default_factory=list, description="Any critical environment-level alerts or warnings.")
    classified: int = Field(description="Count mapping of successfully assigned categories.")
    prioritized: int = Field(description="Count mapping of successfully assigned priorities.")
    routed: int = Field(description="Count mapping of successfully routed emails.")

class TaskConfig(BaseModel):
    """Configuration mapping configuring difficulty and randomization."""
    num_emails: int = Field(description="Total quantity of emails to generate.")
    time_budget: int = Field(description="Length constraint imposed upon agent lifespan.")
    seed: int = Field(description="Random initialization seed ensuring deterministic environments.")
    ambiguity_level: float = Field(default=0.0, description="Noise parameter defining missing email context probability.")
    has_phishing: bool = Field(default=False, description="Flag indicating presence of adversarial phishing content.")
    time_pressure: float = Field(default=0.0, description="Multiplicative modifier to step consumption.")

class EpisodeResult(BaseModel):
    """The aggregate end-of-episode simulation final evaluation layout."""
    score: float = Field(description="Final computed grade output index.")
    steps: int = Field(description="Final number of steps consumed.")
    correct_classifications: int = Field(description="Sum of absolute correct category assignments.")
    phishing_detected: int = Field(description="Sum of absolute positive phishing assertions caught.")
    efficiency: float = Field(description="Computed time-efficiency normalized representation.")
    classification_acc: float = Field(default=0.0, description="Normalized 0-1 category assignment grade.")
    priority_acc: float = Field(default=0.0, description="Normalized 0-1 priority assignment grade.")
    routing_acc: float = Field(default=0.0, description="Normalized 0-1 routing assignment grade.")
    risk_score: float = Field(default=0.0, description="Normalized 0-1 risk isolation grade.")
    efficiency_score: float = Field(default=0.0, description="Normalized 0-1 temporal efficiency grade.")