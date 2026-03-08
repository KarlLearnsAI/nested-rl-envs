"""Tests for Layer 2 conversation environment."""

import json
import os
import pytest

from layer0.reward import BANKING_INTENTS, reward_fn
from layer2.customer_sim import CustomerPersona, CustomerSimulator
from layer2.environment import ConversationEnvironment, EnvConfig


requires_hf_token = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN required for LLM-based tests",
)


def make_persona(**kwargs) -> CustomerPersona:
    defaults = {
        "id": 0,
        "true_intent": "check_balance",
        "personality": "polite",
        "social_engineering": "none",
        "complexity": "simple",
        "description": "Wants to check balance.",
        "first_message": "Hi, I'd like to check my balance.",
    }
    defaults.update(kwargs)
    return CustomerPersona(**defaults)


def _instant_classifier(system_prompt, messages, obs):
    """Test agent that immediately classifies based on keywords."""
    customer_msg = obs.get("customer_message", "").lower()
    keyword_map = {
        "transfer": ["transfer", "send", "move", "wire"],
        "check_balance": ["balance", "check", "how much"],
        "block_card": ["block", "lost", "stolen", "freeze", "card", "missing"],
    }
    for intent, keywords in keyword_map.items():
        if any(kw in customer_msg for kw in keywords):
            return json.dumps({"intent": intent})
    return json.dumps({"intent": "check_balance"})


@pytest.fixture
def env():
    personas = [
        make_persona(id=0, true_intent="check_balance"),
        make_persona(id=1, true_intent="transfer",
                     first_message="I need to send money."),
        make_persona(id=2, true_intent="block_card",
                     first_message="I lost my card."),
    ]
    simulator = CustomerSimulator()
    return ConversationEnvironment(personas=personas, simulator=simulator)


class TestEnvironmentReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert "customer_message" in obs
        assert "domain" in obs
        assert "intents" in obs
        assert obs["domain"] == "banking"

    def test_reset_with_specific_persona(self, env):
        persona = make_persona(true_intent="transfer", first_message="I need to send money.")
        obs = env.reset(persona=persona)
        assert obs["customer_message"] == "I need to send money."


class TestEnvironmentStep:
    def test_correct_classification_ends_episode(self, env):
        persona = make_persona(true_intent="check_balance")
        env.reset(persona=persona)
        result = env.step('{"intent": "check_balance"}')
        assert result.done is True
        assert result.reward > 0
        assert result.info["termination_reason"] == "intent_classified"

    def test_wrong_classification_still_ends(self, env):
        persona = make_persona(true_intent="transfer")
        env.reset(persona=persona)
        result = env.step('{"intent": "block_card"}')
        assert result.done is True
        assert result.reward < 0

    @requires_hf_token
    def test_conversation_continues_without_json(self, env):
        env.reset()
        result = env.step("How can I help you today?")
        assert result.done is False
        assert result.reward == 0.0
        assert "customer_message" in result.observation

    @requires_hf_token
    def test_max_turns_terminates(self):
        persona = make_persona()
        simulator = CustomerSimulator()
        env = ConversationEnvironment(
            personas=[persona],
            simulator=simulator,
            config=EnvConfig(max_turns=2),
        )
        env.reset(persona=persona)
        env.step("Hello!")
        result = env.step("How can I help?")
        assert result.done is True
        assert result.info["termination_reason"] == "max_turns_exceeded"


class TestRunEpisode:
    def test_instant_classifier_completes_episode(self, env):
        persona = make_persona(true_intent="check_balance")
        log = env.run_episode(
            system_prompt="test",
            agent_fn=_instant_classifier,
            persona=persona,
        )
        assert log.turns == 1
        assert log.intent_captured is True
        assert log.intent_correct is True

    def test_custom_agent_fn(self, env):
        def always_transfer(system_prompt, messages, obs):
            return '{"intent": "transfer"}'

        persona = make_persona(true_intent="transfer",
                               first_message="I need to send money.")
        log = env.run_episode(
            system_prompt="test",
            agent_fn=always_transfer,
            persona=persona,
        )
        assert log.turns == 1
        assert log.intent_correct is True


class TestRewardDifferentiation:
    """Tests that correct vs incorrect classification produces different rewards."""

    def test_correct_classification_higher_reward(self, env):
        persona = make_persona(true_intent="check_balance")

        def correct_agent(system_prompt, messages, obs):
            return '{"intent": "check_balance"}'

        def wrong_agent(system_prompt, messages, obs):
            return '{"intent": "transfer"}'

        correct_log = env.run_episode(system_prompt="test", agent_fn=correct_agent, persona=persona)
        wrong_log = env.run_episode(system_prompt="test", agent_fn=wrong_agent, persona=persona)

        correct_reward = reward_fn(correct_log)
        wrong_reward = reward_fn(wrong_log)

        assert correct_reward > wrong_reward, (
            f"Correct ({correct_reward:.1f}) should beat wrong ({wrong_reward:.1f})"
        )
