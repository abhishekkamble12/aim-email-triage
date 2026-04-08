"""AIMEnv — OpenEnv-compliant email triage RL environment.

Dense reward structure (all values are per-step scalars):

  Step penalty          -0.02   every step (discourages looping)
  Open email            +0.01   first open of an email
  Correct classify      +0.35
  Correct priority      +0.20   only awarded when category is also correct
  Correct route         +0.20   only awarded when category is also correct
  Wrong classify        -0.25
  Wrong priority        -0.10   only penalised when category is correct
  Wrong route           -0.10   only penalised when category is correct
  Phishing detected     +0.60   non-critical phishing email
  Phishing critical     +1.00   critical phishing email
  False positive        -0.20   flagged a clean email as phishing
  No-open penalty       -0.15   tried to classify without opening first
  Already processed     -0.05   acted on a done email
  Invalid action        -0.05   unknown action type or missing fields
  Timeout               -0.50   time budget exhausted
  Delay penalty         -0.30   per-step penalty for ignoring a critical email

Loop-prevention: the step penalty (-0.02) accumulates every turn, making
repeated no-ops strictly worse than submitting. The phishing bonus (+1.00)
is large enough to dominate a full episode of step penalties, so the agent
is always incentivised to detect phishing before submitting.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .email_generator import EmailGenerator
from .models import (
    Action,
    EmailFull,
    EmailGroundTruth,
    EmailPartial,
    EpisodeResult,
    Observation,
    TaskConfig,
)
from .reward import Reward

# ---------------------------------------------------------------------------
# Reward constants — single source of truth
# ---------------------------------------------------------------------------

_R_STEP: float = -0.02
_R_OPEN: float = 0.01
_R_CORRECT_CLASS: float = 0.35
_R_CORRECT_PRI: float = 0.20
_R_CORRECT_ROUTE: float = 0.20
_R_WRONG_CLASS: float = -0.25
_R_WRONG_PRI: float = -0.10
_R_WRONG_ROUTE: float = -0.10
_R_PHISHING: float = 0.60
_R_PHISHING_CRITICAL: float = 1.00
_R_FALSE_POSITIVE: float = -0.20
_R_NO_OPEN: float = -0.15
_R_ALREADY_PROCESSED: float = -0.05
_R_INVALID: float = -0.05
_R_TIMEOUT: float = -0.50
_R_DELAY: float = -0.30


class AIMEnv:
    """OpenEnv-compliant email triage environment.

    Args:
        config: Task configuration (seed, email count, time budget, etc.).
    """

    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        self.generator = EmailGenerator(config.seed)

        # Episode state — initialised properly in reset()
        self.emails: List[EmailFull] = []
        self.ground_truths: Dict[str, EmailGroundTruth] = {}
        self.opened: set[str] = set()
        self.time_left: int = config.time_budget
        self.step_count: int = 0
        self.current_obs: Optional[Observation] = None
        self.done: bool = False
        self.pending_penalties: Dict[str, float] = {}
        self.email_status: Dict[str, str] = {}  # unread | opened | processed
        self.penalized_emails: set[str] = set()

        # Accuracy counters
        self.correct_classifications: int = 0
        self.correct_priorities: int = 0
        self.correct_routes: int = 0
        self.phishing_detected: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self.emails, truths_list = self.generator.generate_emails(
            self.config.num_emails,
            has_phishing=self.config.has_phishing,
            ambiguity_level=self.config.ambiguity_level,
        )
        self.ground_truths = {t.id: t for t in truths_list}
        self.opened = set()
        self.time_left = self.config.time_budget
        self.step_count = 0
        self.done = False
        self.pending_penalties = {}
        self.email_status = {e.id: "unread" for e in self.emails}
        self.penalized_emails = set()
        self.correct_classifications = 0
        self.correct_priorities = 0
        self.correct_routes = 0
        self.phishing_detected = 0
        self.current_obs = self._build_observation()
        return self.current_obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool]:
        """Advance the environment by one step.

        Args:
            action: The agent's chosen action.

        Returns:
            A (observation, reward, done) triple.

        Raises:
            RuntimeError: If called after the episode has ended.
        """
        if self.done:
            raise RuntimeError("Episode has ended — call reset() to start a new one.")

        components: Dict[str, float] = {"step": _R_STEP}

        # Apply any pending delay penalties from the previous step
        for eid, penalty in list(self.pending_penalties.items()):
            components[f"delay_{eid}"] = penalty
            self.penalized_emails.add(eid)
        self.pending_penalties.clear()

        # Dispatch action
        if action.type == "open":
            self._handle_open(action, components)
        elif action.type == "classify":
            self._handle_classify(action, components)
        elif action.type == "detect_phishing":
            self._handle_detect_phishing(action, components)
        elif action.type == "submit":
            self.done = True
            components["submit"] = 0.0
        else:
            components["invalid_action"] = _R_INVALID

        # Queue delay penalties for unprocessed critical emails
        for e in self.emails:
            if (
                self.email_status.get(e.id) == "unread"
                and self.ground_truths[e.id].true_priority == "critical"
                and e.id not in self.penalized_emails
            ):
                self.pending_penalties[e.id] = _R_DELAY

        # Advance time (time_pressure scales the cost per step)
        self.step_count += 1
        step_cost = max(1, round(1 + self.config.time_pressure))
        self.time_left -= step_cost

        # Terminal conditions
        all_processed = all(s == "processed" for s in self.email_status.values())
        if self.time_left <= 0 or all_processed:
            self.done = True
            if self.time_left <= 0:
                components["timeout"] = _R_TIMEOUT

        reward = Reward(value=sum(components.values()), components=components)
        self.current_obs = self._build_observation()
        return self.current_obs, reward, self.done

    def state(self) -> Observation:
        """Return the current observation without advancing the episode."""
        if self.current_obs is None:
            raise RuntimeError("Call reset() before state().")
        return self.current_obs

    def get_result(self) -> EpisodeResult:
        """Compute and return the final episode result."""
        total_emails = max(1, len(self.emails))
        total_phishing = max(
            1, sum(1 for t in self.ground_truths.values() if t.is_phishing)
        )
        eff = max(0.0, 1.0 - self.step_count / max(1, self.config.time_budget * 2))

        return EpisodeResult(
            score=0.0,  # Grader computes the weighted final score
            steps=self.step_count,
            correct_classifications=self.correct_classifications,
            phishing_detected=self.phishing_detected,
            efficiency=eff,
            classification_acc=self.correct_classifications / total_emails,
            priority_acc=self.correct_priorities / total_emails,
            routing_acc=self.correct_routes / total_emails,
            risk_score=self.phishing_detected / total_phishing,
            efficiency_score=eff,
        )

    def get_score(self) -> float:
        """Convenience method: grade the current episode and return the score."""
        from .grader import Grader

        return Grader().grade_episode(self.get_result())  # type: ignore[no-untyped-call]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        unread = [
            EmailPartial(id=e.id, subject=e.subject, sender=e.sender, preview=e.preview)
            for e in self.emails
            if self.email_status[e.id] == "unread"
        ][:5]

        return Observation(
            inbox=unread,
            opened=list(self.opened),
            time_left=self.time_left,
            step_count=self.step_count,
            pending_emails=sum(
                1 for s in self.email_status.values() if s != "processed"
            ),
            alerts=[],
            classified=self.correct_classifications,
            prioritized=self.correct_priorities,
            routed=self.correct_routes,
        )

    def _handle_open(self, action: Action, components: Dict[str, float]) -> None:
        eid = action.email_id
        if not eid or eid not in self.email_status:
            components["invalid_action"] = _R_INVALID
        elif self.email_status[eid] == "processed":
            components["already_processed"] = _R_ALREADY_PROCESSED
        elif eid in self.opened:
            components["already_opened"] = _R_ALREADY_PROCESSED
        else:
            self.opened.add(eid)
            self.email_status[eid] = "opened"
            components["opened"] = _R_OPEN

    def _handle_classify(self, action: Action, components: Dict[str, float]) -> None:
        eid = action.email_id
        if not eid or not action.category:
            components["invalid_action"] = _R_INVALID
            return

        if self.email_status.get(eid) != "opened":
            components["no_open_penalty"] = _R_NO_OPEN
            return

        truth = self.ground_truths[eid]
        self.email_status[eid] = "processed"
        self.opened.discard(eid)

        if action.category == truth.true_category:
            components["correct_classification"] = _R_CORRECT_CLASS
            self.correct_classifications += 1

            if action.priority == truth.true_priority:
                components["correct_priority"] = _R_CORRECT_PRI
                self.correct_priorities += 1
            else:
                components["wrong_priority"] = _R_WRONG_PRI

            if action.route == truth.true_route:
                components["correct_route"] = _R_CORRECT_ROUTE
                self.correct_routes += 1
            else:
                components["wrong_route"] = _R_WRONG_ROUTE
        else:
            components["wrong_classification"] = _R_WRONG_CLASS

    def _handle_detect_phishing(
        self, action: Action, components: Dict[str, float]
    ) -> None:
        eid = action.email_id
        if not eid or eid not in self.email_status:
            components["invalid_action"] = _R_INVALID
            return

        truth = self.ground_truths[eid]
        self.email_status[eid] = "processed"
        self.opened.discard(eid)

        if truth.is_phishing:
            if truth.true_priority == "critical":
                components["phishing_detected_critical"] = _R_PHISHING_CRITICAL
            else:
                components["phishing_detected"] = _R_PHISHING
            self.phishing_detected += 1
        else:
            components["false_positive"] = _R_FALSE_POSITIVE
