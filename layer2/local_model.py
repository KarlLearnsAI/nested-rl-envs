"""
Shared local Llama model for Layer 2 inference.

Loads Llama 3.1 8B Instruct once via transformers and provides a
chat_completion-like interface used by both CustomerSimulator and HFAgent,
eliminating HF Inference API dependency and its transient errors.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_shared_instance: LocalLlamaModel | None = None


def get_shared_model(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    hf_token: str | None = None,
    device: str = "auto",
) -> LocalLlamaModel:
    """Get or create the singleton local model instance."""
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = LocalLlamaModel(model_id=model_id, hf_token=hf_token, device=device)
    return _shared_instance


class LocalLlamaModel:
    """
    Local Llama model loaded via transformers.

    Provides a generate() method with the same input format (list of
    message dicts) as the HF Inference API, so callers need minimal changes.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        hf_token: str | None = None,
        device: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading local model: %s", model_id)
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Local model loaded on %s", self.model.device)

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion from a list of chat messages."""
        import torch

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens (exclude the prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
