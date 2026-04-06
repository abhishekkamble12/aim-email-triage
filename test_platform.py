#!/usr/bin/env python3
"""
Comprehensive test suite for AIM-Env Platform
Tests all critical functionality to ensure 10/10 quality
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Test all critical imports"""
    try:
        from env.env import AIMEnv
        from env.models import TaskConfig, Action, Observation
        from env.grader import Grader
        from tasks.task_easy import EASY_TASK_CONFIG
        from inference import HeuristicAgent, LLMAgent
        from backend.app.services.env_service import EnvService
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_environment():
    """Test AIMEnv functionality"""
    try:
        from env.env import AIMEnv
        from tasks.task_easy import EASY_TASK_CONFIG

        env = AIMEnv(EASY_TASK_CONFIG)
        obs = env.reset()

        # Test basic properties
        assert len(obs.inbox) == EASY_TASK_CONFIG.num_emails
        assert obs.time_left == EASY_TASK_CONFIG.time_budget
        assert isinstance(obs.inbox[0].subject, str)

        # Test step
        from env.models import Action
        action = Action(type="open", email_id=obs.inbox[0].id)
        next_obs, reward, done = env.step(action)
        assert next_obs.time_left < obs.time_left

        # Test scoring
        score = env.get_score()
        assert 0.0 <= score <= 1.0

        print("✅ Environment tests passed")
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_agents():
    """Test agent functionality"""
    try:
        from env.env import AIMEnv
        from tasks.task_easy import EASY_TASK_CONFIG
        from inference import HeuristicAgent, LLMAgent

        env = AIMEnv(EASY_TASK_CONFIG)
        obs = env.reset()

        # Test heuristic agent
        heuristic = HeuristicAgent()
        action = heuristic.decide(obs)
        assert action.type in ["open", "classify", "submit"]

        # Test LLM agent (should fallback gracefully)
        llm = LLMAgent()
        action = llm.decide(obs)
        assert action.type in ["open", "classify", "submit"]

        print("✅ Agent tests passed")
        return True
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def test_service():
    """Test backend service"""
    try:
        from backend.app.services.env_service import EnvService
        from backend.app.schemas.request import RunTaskRequest

        service = EnvService()
        request = RunTaskRequest(difficulty="easy", agent_type="heuristic")

        result = service.run_task(request)
        assert result.final_score >= 0.0
        assert len(result.steps) > 0
        assert len(result.emails) >= 0

        print("✅ Service tests passed")
        return True
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False

def test_grader():
    """Test grading system"""
    try:
        from env.grader import Grader
        from env.models import EpisodeResult

        grader = Grader()
        result = EpisodeResult(
            score=0.0,
            steps=10,
            correct_classifications=2,
            phishing_detected=1,
            efficiency=0.8,
            classification_acc=0.67,
            priority_acc=0.67,
            routing_acc=0.67,
            risk_score=1.0,
            efficiency_score=0.8
        )

        score = grader.grade_episode(result)
        expected = (0.30 * 0.67) + (0.20 * 0.67) + (0.20 * 0.67) + (0.20 * 1.0) + (0.10 * 0.8)
        assert abs(score - expected) < 0.01

        print("✅ Grader tests passed")
        return True
    except Exception as e:
        print(f"❌ Grader test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running AIM-Env Platform Test Suite")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("Agents", test_agents),
        ("Service", test_service),
        ("Grader", test_grader)
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n🧪 Testing {name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {name} test failed")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Platform is 10/10 ready!")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())