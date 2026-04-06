import sys
sys.path.append('.')
import os
import json
import re
import logging
from env import AIMEnv, Action, Grader
from tasks import EASY_TASK_CONFIG, MEDIUM_TASK_CONFIG, HARD_TASK_CONFIG
import openai

# Configure Production Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aim_env.log', mode='a')
    ]
)
logger = logging.getLogger("AIM_ENV_AGENT")

class LLMAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("No OpenAI API key provided, LLM agent will fallback to heuristic")

    def decide(self, obs) -> Action:
        # If no API key or client, fallback to heuristic logic
        if not self.client:
            logger.info("Using heuristic fallback for LLM agent")
            return self._heuristic_fallback(obs)
        
        inbox_desc = "\n".join([f"- {e.id}: {e.subject} ({e.preview})" for e in obs.inbox])
        prompt = f"""
You are an email triage agent under time pressure.

Inbox:
{inbox_desc}

Opened: {obs.opened}
Time left: {obs.time_left}

You must return a raw JSON object as your action. Valid action types: "open", "classify", "detect_phishing", "submit".
For "open", include "email_id".
For "classify", include "email_id", "category", "priority", "route".
For "detect_phishing", include "email_id".
For "submit", no other fields.

Example:
{{"type": "open", "email_id": "12345"}}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1  # Low temperature for consistent decisions
            )
            action_str = response.choices[0].message.content.strip()
            return self.parse_action(action_str)
        except Exception as e:
            logger.warning(f"LLM decision failed, falling back to heuristic: {str(e)}")
            return self._heuristic_fallback(obs)

    def _heuristic_fallback(self, obs) -> Action:
        """Simple heuristic fallback when LLM is unavailable"""
        if obs.inbox and not obs.opened:
            return Action(type="open", email_id=obs.inbox[0].id)
        elif obs.opened:
            eid = obs.opened[0]
            return Action(
                type="classify", 
                email_id=eid, 
                category="normal", 
                priority="medium", 
                route="inbox"
            )
        else:
            return Action(type="submit")

    def parse_action(self, action_str: str) -> Action:
        # Extract JSON from markdown code blocks if present
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', action_str, re.DOTALL)
        if match:
            action_str = match.group(1)
        
        try:
            data = json.loads(action_str)
            return Action(**data)
        except Exception as e:
            logger.debug(f"Action parse failed: {str(e)}")
            return Action(type="submit")

class HeuristicAgent:
    def __init__(self):
        logger.info("Initializing Heuristic Agent.")

    def decide(self, obs) -> Action:
        if obs.inbox and not obs.opened:
            return Action(type="open", email_id=obs.inbox[0].id)
        elif obs.opened:
            eid = obs.opened[0]
            # Random heuristic rule
            return Action(
                type="classify", 
                email_id=eid, 
                category="normal", 
                priority="medium", 
                route="inbox"
            )
        else:
            return Action(type="submit")

def run_task(config, agent, grader):
    env = AIMEnv(config)
    obs = env.reset()
    
    logs = [f"[START] Num Emails: {config.num_emails}"]
    logger.info(logs[0])
    
    total_reward = 0
    step = 0
    
    while not env.done:
        action = agent.decide(obs)
        obs, reward, done = env.step(action)
        total_reward += reward.value
        
        step_log = f"[STEP {step}] Action: {action.type} -> Reward Component Sum: {reward.value:.2f}"
        logs.append(step_log)
        logger.debug(step_log)
        
        step += 1
        
    result = env.get_result()
    final_score = grader.grade_episode(result, logs)
    
    end_log = f"[END] Final Score: {final_score:.2f} | Efficiency: {result.efficiency:.2f}"
    logs.append(end_log)
    logger.info(end_log)
    
    return final_score

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("OPENAI_API_KEY detected. Using LLMAgent.")
        agent = LLMAgent(api_key)
    else:
        logger.warning("OPENAI_API_KEY not set. Falling back to HeuristicAgent baseline.")
        agent = HeuristicAgent()
        
    grader = Grader()
    tasks_to_run = [EASY_TASK_CONFIG, MEDIUM_TASK_CONFIG, HARD_TASK_CONFIG]
    
    for task_cfg in tasks_to_run:
        logger.info(f"--- Launching Task (Seed: {task_cfg.seed}) ---")
        run_task(task_cfg, agent, grader)