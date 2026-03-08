"""Tests for Layer 2 conversation environment."""

import json
import pytest

from layer0.reward import BANKING_INTENTS
from layer2.customer_sim import CustomerPersona, CustomerSimulator
from layer2.environment import ConversationEnvironment, EnvConfig


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


@pytest.fixture
def env():
    personas = [
        make_persona(id=0, true_intent="check_balance"),
        make_persona(id=1, true_intent="transfer"),
        make_persona(id=2, true_intent="block_card"),
    ]
    simulator = CustomerSimulator()  # rule-based fallback
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
        assert result.reward < 0  # wrong intent is penalized

    def test_conversation_continues_without_json(self, env):
        env.reset()
        result = env.step("How can I help you today?")
        assert result.done is False
        assert result.reward == 0.0
        assert "customer_message" in result.observation

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
    def test_default_agent_completes_episode(self, env):
        log = env.run_episode(system_prompt="You are a helpful agent.")
        assert log.turns > 0
        assert log.intent_captured is True

    def test_custom_agent_fn(self, env):
        def instant_classifier(system_prompt, messages, obs):
            return '{"intent": "check_balance"}'

        persona = make_persona(true_intent="check_balance")
        log = env.run_episode(
            system_prompt="test",
            agent_fn=instant_classifier,
            persona=persona,
        )
        assert log.turns == 1
        assert log.intent_correct is True
