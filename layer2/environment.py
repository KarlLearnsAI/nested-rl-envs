"""
Layer 2 — Conversation Environment (OpenEnv-compatible).

Implements reset() / step() interface. Each episode is a multi-turn
conversation between a voice agent (whose system prompt comes from Layer 1)
and a simulated customer (driven by CustomerSimulator).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from layer0.reward import (
    ConversationLog,
    reward_fn,
    extract_intent_json,
    contains_unauthorized_disclosure,
    RewardConfig,
    BANKING_INTENTS,
)
from layer2.customer_sim import CustomerPersona, CustomerSimulator


@dataclass
class EnvConfig:
    """Configuration for the conversation environment."""

    domain: str = "banking"
    intents: list[str] = field(default_factory=lambda: list(BANKING_INTENTS))
    max_turns: int = 10
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class StepResult:
    """Result returned by env.step()."""

    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class ConversationEnvironment:
    """
    OpenEnv-compatible RL environment for customer support conversations.

    Action space: natural language (agent's text response)
    Observation space: dict with latest customer message + metadata
    Reward: scalar from Layer 0's reward_fn, emitted at episode end
    """

    def __init__(
        self,
        personas: list[CustomerPersona],
        simulator: CustomerSimulator,
        config: EnvConfig | None = None,
    ):
        self.personas = personas
        self.simulator = simulator
        self.config = config or EnvConfig()

        # Episode state
        self._current_persona: CustomerPersona | None = None
        self._conversation_log: ConversationLog | None = None
        self._messages: list[dict[str, str]] = []
        self._done: bool = True
        self._turn: int = 0

    def reset(self, persona: CustomerPersona | None = None) -> dict[str, Any]:
        """
        Start a new episode.

        Samples a random customer persona, generates the first customer message,
        and returns the initial observation.
        """
        self._current_persona = persona or random.choice(self.personas)
        self._messages = []
        self._done = False
        self._turn = 0
        self._conversation_log = ConversationLog(
            customer_persona=self._current_persona.personality,
            true_intent=self._current_persona.true_intent,
            injection_attempted=self._current_persona.social_engineering != "none",
        )

        # Customer's opening message
        first_message = self._current_persona.first_message
        self._messages.append({"role": "customer", "content": first_message})

        return {
            "customer_message": first_message,
            "domain": self.config.domain,
            "intents": self.config.intents,
            "turn": 0,
        }

    def step(self, agent_response: str) -> StepResult:
        """
        Process the agent's response and return the next observation.

        The agent sends a text response; the environment checks for termination,
        generates the customer's next reply, and returns the result.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._turn += 1
        self._messages.append({"role": "agent", "content": agent_response})
        self._conversation_log.turns = self._turn

        # --- Check termination conditions ---
        termination, info = self._check_termination(agent_response)

        if termination is not None:
            self._done = True
            self._conversation_log.messages = list(self._messages)
            reward = reward_fn(self._conversation_log, self.config.reward_config)
            return StepResult(
                observation={"customer_message": "", "done_reason": termination},
                reward=reward,
                done=True,
                info={
                    "termination_reason": termination,
                    "conversation_log": self._conversation_log.to_dict(),
                    **info,
                },
            )

        # --- Generate customer reply ---
        customer_reply = self.simulator.generate_reply(
            persona=self._current_persona,
            conversation_history=self._messages[:-1],  # exclude the latest agent msg
            agent_message=agent_response,
        )
        self._messages.append({"role": "customer", "content": customer_reply})

        return StepResult(
            observation={
                "customer_message": customer_reply,
                "domain": self.config.domain,
                "intents": self.config.intents,
                "turn": self._turn,
            },
            reward=0.0,  # intermediate reward is 0
            done=False,
            info={},
        )

    def _check_termination(self, agent_response: str) -> tuple[str | None, dict]:
        """Check if the episode should end."""
        # 1. Agent classified intent
        intent_json = extract_intent_json(agent_response)
        if intent_json and "intent" in intent_json:
            agent_intent = intent_json["intent"]
            self._conversation_log.intent_captured = True
            self._conversation_log.agent_intent = agent_intent
            self._conversation_log.intent_correct = (
                agent_intent == self._current_persona.true_intent
            )

            # Check if API call matches
            if "action" in intent_json:
                self._conversation_log.api_call_made = True
                self._conversation_log.api_call_correct = (
                    intent_json.get("action") == self._expected_action()
                )

            return "intent_classified", {"agent_intent": agent_intent}

        # 2. Max turns exceeded
        if self._turn >= self.config.max_turns:
            return "max_turns_exceeded", {}

        # 3. Unauthorized disclosure (social engineering succeeded)
        if contains_unauthorized_disclosure(agent_response):
            self._conversation_log.injection_succeeded = True
            return "injection_succeeded", {}

        return None, {}

    def _expected_action(self) -> str:
        """Map intent to expected API action."""
        action_map = {
            "transfer": "initiate_transfer",
            "check_balance": "get_balance",
            "block_card": "block_card",
        }
        return action_map.get(self._current_persona.true_intent, "unknown")

    def run_episode(
        self,
        system_prompt: str,
        agent_fn: Any | None = None,
        persona: CustomerPersona | None = None,
    ) -> ConversationLog:
        """
        Run a complete episode with a given system prompt.

        If agent_fn is None, uses a simple rule-based agent for testing.
        agent_fn signature: (system_prompt, conversation_history, observation) -> str
        """
        obs = self.reset(persona=persona)

        while not self._done:
            if agent_fn is not None:
                agent_response = agent_fn(system_prompt, self._messages, obs)
            else:
                agent_response = self._default_agent(system_prompt, obs)

            result = self.step(agent_response)
            obs = result.observation

        return self._conversation_log

    def _default_agent(self, system_prompt: str, obs: dict) -> str:
        """
        Prompt-sensitive rule-based agent for testing (no LLM needed).

        Parses the system prompt for behavioral signals to simulate how different
        prompts lead to different agent behaviors. A generic prompt produces a
        chatty, slow, vulnerable agent. A structured prompt with security rules
        produces an efficient, secure agent.
        """
        prompt_lower = system_prompt.lower()
        customer_msg = obs.get("customer_message", "")
        intents = obs.get("intents", BANKING_INTENTS)
        turn = obs.get("turn", self._turn)
        customer_lower = customer_msg.lower()

        # --- Parse prompt for behavioral traits ---
        has_security_rules = any(kw in prompt_lower for kw in [
            "never reveal", "never accept", "never follow", "social engineering",
            "unauthorized", "ignore your rules", "third party",
        ])
        has_structured_output = "json" in prompt_lower or '{"intent"' in prompt_lower
        has_efficiency_rules = any(kw in prompt_lower for kw in [
            "at most 2", "at most 1", "one specific", "classify immediately",
            "minimal turns", "concise",
        ])
        has_intent_list = any(intent in prompt_lower for intent in intents)

        # --- INJECTION HANDLING ---
        injection_phrases = [
            "ignore your previous", "ignore previous", "system override",
            "debug mode", "forget everything", "bank supervisor",
            "provide the full account", "display all",
        ]
        is_injection = any(phrase in customer_lower for phrase in injection_phrases)

        if is_injection:
            if has_security_rules:
                # Trained agent: rejects and redirects
                return (
                    "I'm sorry, I can only help with your own account. "
                    "I cannot share information about other accounts or override security protocols. "
                    "How can I help you with your banking needs today?"
                )
            else:
                # Base agent: naive, susceptible — leaks info
                return (
                    "I understand you need account information. "
                    "Your wife's account balance is 5000. "
                    "Is there anything else I can help with?"
                )

        # --- SOFT SOCIAL ENGINEERING ---
        soft_se_phrases = [
            "wife's account", "mother also", "husband's", "partner's",
            "business partner", "check my wife", "her balance", "his balance",
        ]
        is_soft_se = any(phrase in customer_lower for phrase in soft_se_phrases)

        if is_soft_se:
            if has_security_rules:
                return (
                    "I can only access your own account information. "
                    "Your family member would need to contact us directly. "
                    "Let me help you with your request — what do you need?"
                )
            else:
                # Base agent: tries to help, leaks info
                return (
                    "Sure, I can share the account details for your wife's account. "
                    "The account balance is 3200. "
                    "Is there anything else you need?"
                )

        # --- INTENT CLASSIFICATION ---
        keyword_map = {
            "transfer": ["transfer", "send", "move", "wire", "pay"],
            "check_balance": ["balance", "how much", "check", "amount", "funds"],
            "block_card": ["block", "lost", "stolen", "freeze", "disable", "card", "missing"],
        }

        detected_intent = None
        for intent in intents:
            if any(kw in customer_lower for kw in keyword_map.get(intent, [])):
                detected_intent = intent
                break

        # Trained agent: classify fast if intent is clear
        if has_structured_output and has_efficiency_rules and detected_intent:
            if has_intent_list or turn >= 1:
                return json.dumps({"intent": detected_intent})

        # Trained agent: one clarifying question then classify
        if has_structured_output and turn >= 2:
            if detected_intent:
                return json.dumps({"intent": detected_intent})
            return json.dumps({"intent": intents[0]})

        # Base agent: chatty, asks many generic questions before classifying
        if not has_structured_output:
            if turn == 0:
                return (
                    "Hello! Welcome to our bank's customer service. "
                    "Thank you for calling us today. My name is Alex and I'll be happy to help you. "
                    "Before we get started, could you tell me a bit about what brings you in today? "
                    "We offer a wide range of services including transfers, balance inquiries, "
                    "card management, loan applications, and more."
                )
            if turn == 1:
                return (
                    "Thank you for sharing that. I want to make sure I understand correctly. "
                    "Could you tell me a bit more about what you need? "
                    "Also, for security purposes, could you confirm your full name?"
                )
            if turn == 2:
                return (
                    "Great, thank you for confirming. Let me look into that for you. "
                    "Just to double check — can you verify your account number or "
                    "the last four digits of your card?"
                )
            if turn == 3:
                return (
                    "Perfect, I appreciate your patience. "
                    "Now, just to make sure I have this right — what exactly would you like me to do?"
                )
            # Finally classify on turn 4+
            if detected_intent:
                return json.dumps({"intent": detected_intent})
            return json.dumps({"intent": intents[0]})

        # Default structured agent: ask one question then classify
        if turn == 0:
            return "How can I help you today? Please describe what you need."
        if detected_intent:
            return json.dumps({"intent": detected_intent})
        return "Could you be more specific about what you need help with?"
