"""
Voice agent for Layer 2 conversations.

Uses Llama 3.1 8B Instruct to act as the customer support agent during
evaluation. Supports both local model and HF Inference API backends.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None  # type: ignore

logger = logging.getLogger(__name__)


class HFAgent:
    """
    Voice agent powered by Llama 3.1 8B.

    Takes a system prompt from Layer 1 and generates responses
    in the customer support conversation.

    Supports two backends:
    - local: loads model in-process via transformers (pass local_model=...)
    - api: uses HF Inference API (pass hf_token=...)
    """

    DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    def __init__(
        self,
        model_id: str | None = None,
        hf_token: str | None = None,
        max_tokens: int = 300,
        temperature: float = 0.3,
        local_model: Any = None,
    ):
        self.model_id = model_id or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._local_model = local_model
        self._client: Any = None

        if local_model is None:
            self.hf_token = hf_token or os.environ.get("HF_TOKEN")
            if self.hf_token and InferenceClient is not None:
                self._client = InferenceClient(token=self.hf_token)

    @property
    def is_llm_available(self) -> bool:
        return self._local_model is not None or self._client is not None

    def __call__(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        observation: dict[str, Any],
        max_retries: int = 4,
    ) -> str:
        """
        Generate an agent response.

        Compatible with ConversationEnvironment.run_episode(agent_fn=...).
        """
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

        if self._local_model is not None:
            return self._local_model.generate(
                messages, max_tokens=self.max_tokens, temperature=self.temperature,
            )

        if self._client is None:
            raise RuntimeError(
                "No inference backend available. "
                "Pass local_model=... or set HF_TOKEN for API access."
            )

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat_completion(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e)
                if "402" in err_str or "Payment Required" in err_str:
                    raise RuntimeError(
                        "HF API credits depleted. "
                        "Get more credits at https://huggingface.co/settings/billing"
                    ) from e
                if any(code in err_str for code in ("500", "502", "503", "504", "429", "timeout", "Timeout", "Time-out")):
                    last_err = e
                    wait = 2 ** (attempt + 1)  # 2, 4, 8, 16s
                    logger.warning(
                        "HF API error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, max_retries + 1, wait, e,
                    )
                    time.sleep(wait)
                    continue
                raise
        raise last_err  # type: ignore[misc]
