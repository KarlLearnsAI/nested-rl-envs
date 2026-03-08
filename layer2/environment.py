"""
Layer 2 — Conversation Environment (OpenEnv-compatible).

Implements reset() / step() interface. Each episode is a multi-turn
conversation between a voice agent (whose system prompt comes from Layer 1)
and a simulated customer (driven by CustomerSimulator).
"""

from __future__ import annotations

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
        agent_fn: Any,
        persona: CustomerPersona | None = None,
    ) -> ConversationLog:
        """
        Run a complete episode with a given system prompt.

        agent_fn signature: (system_prompt, conversation_history, observation) -> str
        """
        obs = self.reset(persona=persona)

        while not self._done:
            agent_response = agent_fn(system_prompt, self._messages, obs)
            result = self.step(agent_response)
            obs = result.observation

        return self._conversation_log
