import sys
sys.path.append('..')
from env.env import AIMEnv
from tasks import EASY_TASK_CONFIG, MEDIUM_TASK_CONFIG, HARD_TASK_CONFIG
from env.models import Action
import random
import openai
import os
import json

class LLMAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
        self.experience = []  # For few-shot learning
        self.q = {}  # Q-values for RL
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.9  # Start high for exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95

    def get_state_key(self, obs):
        return (tuple(obs.opened), obs.time_left)

    def decide(self, obs):
        state = self.get_state_key(obs)
        if random.random() < self.epsilon:
            return self.random_action(obs)

        # LLM decision with experience
        prompt = self.build_prompt(obs)
        if self.client:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            action_str = response.choices[0].message.content.strip()
        else:
            # Mock LLM behavior: pick action with highest Q-value, or fallback.
            q_values = self.q.get(state, {})
            if q_values:
                best_action_type = max(q_values, key=q_values.get)
                if best_action_type == "open" and obs.inbox:
                    action_str = f"open {obs.inbox[0].id}"
                else:
                    action_str = "submit"
            else:
                action_str = "submit"
        
        return self.parse_action(action_str, obs)

    def build_prompt(self, obs):
        # Few-shot from experience
        few_shots = "\n".join([f"State: {e['state']}\nAction: {e['action']}\nReward: {e['reward']}" for e in self.experience[-3:]])
        inbox_desc = "\n".join([f"- {e.subject} ({e.preview[:50]}...)" for e in obs.inbox])
        
        state = self.get_state_key(obs)
        q_vals_str = "\n".join([f"{a}: {v:.2f}" for a, v in self.q.get(state, {}).items()])
        if not q_vals_str:
            q_vals_str = "None (Exploring)"
            
        return f"""
You are an email triage agent. Maximize reward over time.

Inbox:
{inbox_desc}

Opened: {obs.opened}
Time left: {obs.time_left}

Known Q-Values for current state (higher is better):
{q_vals_str}

Past successful actions:
{few_shots}

Choose action and Output ONLY valid JSON in one of these formats:
{{"type": "open", "email_id": "<id>"}}
{{"type": "classify", "email_id": "<id>", "category": "<cat>", "priority": "<pri>", "route": "<route>"}}
{{"type": "detect_phishing", "email_id": "<id>"}}
{{"type": "submit"}}
"""

    def parse_action(self, action_str, obs):
        try:
            # Clean possible markdown bounds
            action_str = action_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(action_str)
            return Action(
                type=data.get("type", "submit"),
                email_id=data.get("email_id"),
                category=data.get("category"),
                priority=data.get("priority"),
                route=data.get("route")
            )
        except Exception:
            return Action(type="submit")  # Safe fallback

    def random_action(self, obs):
        if obs.inbox:
            email = random.choice(obs.inbox)
            return Action(type="open", email_id=email.id)
        return Action(type="submit")

    def update(self, state, action, reward, next_state):
        # Q-learning update
        if state not in self.q:
            self.q[state] = {}
        if next_state not in self.q:
            self.q[next_state] = {}
            
        current_q = self.q[state].get(action.type, 0.0)
        next_max = max(self.q[next_state].values(), default=0.0)
        
        self.q[state][action.type] = current_q + self.alpha * (reward + self.gamma * next_max - current_q)

    def learn_from_episode(self, episode_log):
        # Add to experience only if positive reward
        for step in episode_log:
            if step["reward"] > 0:
                self.experience.append(step)
        # Keep last 50
        self.experience = self.experience[-50:]
        
        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY not set. Running in local MOCK mode.")

    agent = LLMAgent(api_key)
    tasks = [EASY_TASK_CONFIG, MEDIUM_TASK_CONFIG, HARD_TASK_CONFIG]
    scores = []

    for episode in range(100):
        task = random.choice(tasks)
        env = AIMEnv(task)
        obs = env.reset()
        total_reward = 0
        episode_log = []

        while not env.done:
            state = agent.get_state_key(obs)
            action = agent.decide(obs)
            next_obs, reward, done = env.step(action)
            next_state = agent.get_state_key(next_obs)
            agent.update(state, action, reward.value, next_state)
            episode_log.append({"state": str(state), "action": str(action.type), "reward": reward.value})
            obs = next_obs
            total_reward += reward.value

        agent.learn_from_episode(episode_log)
        scores.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}: {total_reward:.2f} (avg last 10: {sum(scores[-10:])/10:.2f})")

    with open("scores.json", "w") as f:
        json.dump(scores, f)
    print("Training complete. Scores improved from", scores[0], "to", scores[-1])

if __name__ == "__main__":
    train()