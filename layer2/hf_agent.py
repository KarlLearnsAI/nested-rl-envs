"""
HF Inference API wrapper for the voice agent (Layer 2).

Uses a small model via HF Inference to act as the customer support agent
during evaluation. In training (Layer 1), the agent is the model being optimized.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None  # type: ignore


class HFAgent:
    """
    Voice agent powered by HF Inference API.

    This wraps a small model (e.g. Qwen 2.5 3B) with a system prompt
    from Layer 1, and generates responses in the customer support conversation.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, model_id: str | None = None, hf_token: str | None = None):
        self.model_id = model_id or self.DEFAULT_MODEL
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._client: Any = None
        if self.hf_token and InferenceClient is not None:
            self._client = InferenceClient(token=self.hf_token)

    def __call__(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        observation: dict[str, Any],
    ) -> str:
        """
        Generate an agent response.

        Compatible with ConversationEnvironment.run_episode(agent_fn=...).
        """
        if self._client is None:
            return self._fallback_response(observation)

        messages = [{"role": "system", "content": system_prompt}]

        for msg in conversation_history:
            if msg["role"] == "customer":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "agent":
                messages.append({"role": "assistant", "content": msg["content"]})

        # Add the latest customer message from observation
        customer_msg = observation.get("customer_message", "")
        if customer_msg:
            messages.append({"role": "user", "content": customer_msg})

        response = self._client.chat_completion(
            model=self.model_id,
            messages=messages,
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _fallback_response(self, observation: dict[str, Any]) -> str:
        """Rule-based fallback when no HF token is available."""
        customer_msg = observation.get("customer_message", "").lower()
        intents = observation.get("intents", [])

        keywords = {
            "transfer": ["transfer", "send", "move", "wire", "pay"],
            "check_balance": ["balance", "how much", "check", "amount", "funds"],
            "block_card": ["block", "lost", "stolen", "freeze", "disable", "card"],
        }

        for intent in intents:
            if any(kw in customer_msg for kw in keywords.get(intent, [])):
                return json.dumps({"intent": intent})

        turn = observation.get("turn", 0)
        if turn >= 2:
            return json.dumps({"intent": intents[0] if intents else "unknown"})

        return "Could you please describe what you need help with today?"
