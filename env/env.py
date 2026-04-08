from typing import Optional, Dict, List
from .models import Action, Observation, TaskConfig, EmailPartial, EmailFull, EpisodeResult, EmailGroundTruth
from .reward import Reward
from .email_generator import EmailGenerator

class AIMEnv:
    def __init__(self, config: TaskConfig):
        self.config = config
        self.generator = EmailGenerator(config.seed)
        self.emails: List[EmailFull] = []
        self.ground_truths: Dict[str, EmailGroundTruth] = {}
        self.opened: set[str] = set()
        
        self.time_left = config.time_budget
        self.step_count = 0
        self.current_obs: Optional[Observation] = None
        self.done = False
        
        self.pending_penalties: Dict[str, float] = {}  # Delayed consequences
        self.email_status: Dict[str, str] = {}  # unread, opened, processed, ignored
        
        # Stats
        self.correct_classifications = 0
        self.correct_priorities = 0
        self.correct_routes = 0
        self.phishing_detected = 0
        self.penalized_emails = set()

    def reset(self) -> Observation:
        self.emails, truths_list = self.generator.generate_emails(
            self.config.num_emails, 
            has_phishing=self.config.has_phishing, 
            ambiguity_level=self.config.ambiguity_level
        )
        self.ground_truths = {t.id: t for t in truths_list}
        
        self.opened = set()
        self.time_left = self.config.time_budget
        self.step_count = 0
        self.done = False
        
        self.pending_penalties = {}
        self.email_status = {e.id: "unread" for e in self.emails}
        
        self.correct_classifications = 0
        self.correct_priorities = 0
        self.correct_routes = 0
        self.phishing_detected = 0
        self.penalized_emails = set()
        
        self.current_obs = self._build_observation()
        return self.current_obs

    def _build_observation(self) -> Observation:
        unread_emails = [
            EmailPartial(id=e.id, subject=e.subject, sender=e.sender, preview=e.preview)
            for e in self.emails if self.email_status[e.id] == "unread"
        ][:5]
        
        return Observation(
            inbox=unread_emails,
            opened=list(self.opened),
            time_left=self.time_left,
            step_count=self.step_count,
            pending_emails=sum(1 for s in self.email_status.values() if s != "processed"),
            alerts=[],
            classified=self.correct_classifications,
            prioritized=self.correct_priorities,
            routed=self.correct_routes
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool]:
        if self.done:
            raise RuntimeError("Episode ended, call reset()")

        components = {"step": -0.02}
        
        # Apply delays
        for eid, penalty in list(self.pending_penalties.items()):
            components[f"delay_{eid}"] = penalty
            self.penalized_emails.add(eid)
        self.pending_penalties.clear()

        # Handle Action
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
            components["invalid_action"] = -0.05

        # Add new delayed penalties for ignored critical emails
        for e in self.emails:
            if self.email_status.get(e.id) == "unread" and self.ground_truths[e.id].true_priority == "critical" and e.id not in self.penalized_emails:
                self.pending_penalties[e.id] = -0.3

        self.step_count += 1
        
        time_penalty = 1
        if self.config.time_pressure > 0:
            time_penalty += self.config.time_pressure
        self.time_left -= int(time_penalty)
        
        if self.time_left <= 0 or all(s == "processed" for s in self.email_status.values()):
            self.done = True
            if self.time_left <= 0:
                components["timeout"] = -0.50

        total_value = sum(components.values())
        reward = Reward(value=total_value, components=components)
        
        self.current_obs = self._build_observation()
        return self.current_obs, reward, self.done

    def _handle_open(self, action: Action, components: Dict[str, float]):
        if not action.email_id or action.email_id not in self.email_status:
            components["invalid_action"] = -0.05
        elif self.email_status[action.email_id] == "processed":
            components["already_processed"] = -0.05
        elif action.email_id in self.opened:
            components["already_opened"] = -0.05
        else:
            self.opened.add(action.email_id)
            self.email_status[action.email_id] = "opened"
            components["opened"] = 0.01

    def _handle_classify(self, action: Action, components: Dict[str, float]):
        if not action.email_id or not action.category:
            components["invalid_action"] = -0.05
            return
            
        if self.email_status.get(action.email_id) != "opened":
            components["no_open_penalty"] = -0.15
            return

        truth = self.ground_truths[action.email_id]
        self.email_status[action.email_id] = "processed"
        self.opened.discard(action.email_id)

        if action.category == truth.true_category:
            components["correct_classification"] = 0.35
            self.correct_classifications += 1
            if action.priority == truth.true_priority:
                components["correct_priority"] = 0.20
                self.correct_priorities += 1
            else:
                components["wrong_priority"] = -0.10
                
            if action.route == truth.true_route:
                components["correct_route"] = 0.20
                self.correct_routes += 1
            else:
                components["wrong_route"] = -0.10
        else:
            components["wrong_classification"] = -0.25

    def _handle_detect_phishing(self, action: Action, components: Dict[str, float]):
        if not action.email_id or action.email_id not in self.email_status:
            components["invalid_action"] = -0.05
            return

        truth = self.ground_truths[action.email_id]
        self.email_status[action.email_id] = "processed"
        self.opened.discard(action.email_id)

        if truth.is_phishing:
            if truth.true_priority == "critical":
                components["phishing_detected_critical"] = 1.00
            else:
                components["phishing_detected"] = 0.60
            self.phishing_detected += 1
        else:
            components["false_positive"] = -0.20

    def get_result(self) -> EpisodeResult:
        total_emails = max(1, len(self.emails))
        total_phishing = max(1, sum(1 for t in self.ground_truths.values() if t.is_phishing))
        
        eff = 1.0 - (self.step_count / max(1, self.config.time_budget * 2))
        cls_acc = self.correct_classifications / total_emails
        pri_acc = self.correct_priorities / total_emails
        rte_acc = self.correct_routes / total_emails
        risk = self.phishing_detected / total_phishing
        
        # We will let grader compute the weighted actual score based on the result.
        # So we just pass raw components to EpisodeResult.
        
        return EpisodeResult(
            score=0.0, # Final score will be calculated by the Grader
            steps=self.step_count,
            correct_classifications=self.correct_classifications,
            phishing_detected=self.phishing_detected,
            efficiency=eff,
            classification_acc=cls_acc,
            priority_acc=pri_acc,
            routing_acc=rte_acc,
            risk_score=risk,
            efficiency_score=eff
        )

    def state(self) -> Observation:
        return self.current_obs

    def get_score(self) -> float:
        """Get the final episode score using the Grader"""
        from .grader import Grader
        result = self.get_result()
        grader = Grader()
        return grader.grade_episode(result)