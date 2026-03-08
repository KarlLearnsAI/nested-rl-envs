"""
HF Inference API wrapper for the voice agent (Layer 2).

Uses Llama 3.1 8B Instruct via HF Inference to act as the customer support
agent during evaluation. In training (Layer 1), the agent is the model being
optimized — this module provides the inference-time agent for A/B testing.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None  # type: ignore

logger = logging.getLogger(__name__)


class HFAgent:
    """
    Voice agent powered by HF Inference API.

    Takes a system prompt from Layer 1 and generates responses
    in the customer support conversation using Llama 3.1 8B.
    """

    DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    def __init__(self, model_id: str | None = None, hf_token: str | None = None):
        self.model_id = model_id or self.DEFAULT_MODEL
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._client: Any = None
        if self.hf_token and InferenceClient is not None:
            self._client = InferenceClient(token=self.hf_token)

    @property
    def is_llm_available(self) -> bool:
        return self._client is not None

    def __call__(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        observation: dict[str, Any],
    ) -> str:
        """
        Generate an agent response.

        Compatible with ConversationEnvironment.run_episode(agent_fn=...).
        Requires a valid HF token and working Inference API connection.
        """
        if self._client is None:
            raise RuntimeError(
                "HF Inference API client is not available. "
                "Set HF_TOKEN environment variable with a valid HuggingFace token."
            )

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

        try:
            response = self._client.chat_completion(
                model=self.model_id,
                messages=messages,
                max_tokens=300,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "402" in str(e) or "Payment Required" in str(e):
                raise RuntimeError(
                    "HF API credits depleted. "
                    "Get more credits at https://huggingface.co/settings/billing"
                ) from e
            raise
