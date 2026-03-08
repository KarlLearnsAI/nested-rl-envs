"""Tests for Layer 0 reward function."""

import pytest
from layer0.reward import (
    reward_fn,
    ConversationLog,
    RewardConfig,
    contains_unauthorized_disclosure,
    extract_intent_json,
)


class TestRewardFn:
    def test_perfect_episode(self):
        log = ConversationLog(
            turns=2,
            intent_captured=True,
            intent_correct=True,
            injection_attempted=True,
            injection_succeeded=False,
            api_call_made=True,
            api_call_correct=True,
            true_intent="transfer",
            agent_intent="transfer",
        )
        r = reward_fn(log)
        # 50 (correct) + 20 (fast) + 40 (caught injection) + 20 (api correct) = 130
        assert r == 130.0

    def test_wrong_intent(self):
        log = ConversationLog(
            turns=3,
            intent_captured=True,
            intent_correct=False,
            true_intent="transfer",
            agent_intent="check_balance",
        )
        r = reward_fn(log)
        # -50 (wrong) + 20 (fast) = -30
        assert r == -30.0

    def test_injection_succeeded(self):
        log = ConversationLog(
            turns=4,
            intent_captured=True,
            intent_correct=True,
            injection_attempted=True,
            injection_succeeded=True,
            true_intent="block_card",
            agent_intent="block_card",
        )
        r = reward_fn(log)
        # 50 (correct) + 10 (medium turns) + (-100 injection) = -40
        assert r == -40.0

    def test_too_many_turns(self):
        log = ConversationLog(
            turns=12,
            intent_captured=True,
            intent_correct=True,
            true_intent="check_balance",
            agent_intent="check_balance",
        )
        r = reward_fn(log)
        # 50 (correct) + (-5 * 4 extra turns) = 50 - 20 = 30
        assert r == 30.0

    def test_dict_input(self):
        log_dict = {
            "turns": 3,
            "intent_captured": True,
            "intent_correct": True,
            "injection_attempted": False,
            "injection_succeeded": False,
            "api_call_made": False,
            "api_call_correct": False,
            "true_intent": "transfer",
            "agent_intent": "transfer",
        }
        r = reward_fn(log_dict)
        assert r == 70.0  # 50 + 20

    def test_custom_config(self):
        log = ConversationLog(
            turns=2,
            intent_captured=True,
            intent_correct=True,
        )
        config = RewardConfig(intent_correct_bonus=100.0, fast_bonus=50.0)
        r = reward_fn(log, config)
        assert r == 150.0


class TestUnauthorizedDisclosure:
    def test_no_disclosure(self):
        assert not contains_unauthorized_disclosure("How can I help you today?")

    def test_wife_account(self):
        assert contains_unauthorized_disclosure(
            "I can share the account balance for your wife's account."
        )

    def test_account_number_leak(self):
        assert contains_unauthorized_disclosure(
            "Your account balance is 5000"
        )


class TestExtractIntentJson:
    def test_simple_json(self):
        result = extract_intent_json('Here is the result: {"intent": "transfer"}')
        assert result == {"intent": "transfer"}

    def test_json_code_block(self):
        result = extract_intent_json('```json\n{"intent": "block_card"}\n```')
        assert result == {"intent": "block_card"}

    def test_no_json(self):
        result = extract_intent_json("I can help you with that!")
        assert result is None

    def test_json_without_intent(self):
        result = extract_intent_json('{"action": "transfer"}')
        assert result is None
