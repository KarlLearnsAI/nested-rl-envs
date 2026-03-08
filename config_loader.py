"""
Loads training configuration from config.yaml.

Single source of truth for all training parameters.
CLI arguments override values from the YAML file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from layer0.reward import RewardConfig
from layer2.environment import EnvConfig


_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _ROOT / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the raw YAML config as a dict."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def make_grpo_config(cfg: dict[str, Any]):
    """Build a GRPOConfig from the loaded YAML dict."""
    # Import here to avoid circular imports
    from layer1.grpo_trainer import GRPOConfig

    grpo = cfg.get("grpo", {})
    env = cfg.get("environment", {})
    paths = cfg.get("paths", {})
    gen = cfg.get("generation", {})

    return GRPOConfig(
        model_name=grpo.get("model_name", "unsloth/Qwen2.5-3B-Instruct"),
        lora_r=grpo.get("lora_r", 16),
        lora_alpha=grpo.get("lora_alpha", 16),
        lora_dropout=grpo.get("lora_dropout", 0.0),
        num_candidates=grpo.get("num_candidates", 2),
        episodes_per_candidate=grpo.get("episodes_per_candidate", 3),
        num_training_steps=grpo.get("num_training_steps", 5),
        learning_rate=grpo.get("learning_rate", 5e-5),
        max_prompt_length=grpo.get("max_prompt_length", 2048),
        max_seq_length=gen.get("max_seq_length", 4096),
        prompt_max_new_tokens=gen.get("prompt_max_new_tokens", 2048),
        prompt_temperature=gen.get("prompt_temperature", 0.3),
        per_device_train_batch_size=grpo.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=grpo.get("gradient_accumulation_steps", 4),
        logging_steps=grpo.get("logging_steps", 1),
        save_steps=grpo.get("save_steps", 10),
        sft_warm_start=grpo.get("sft_warm_start", True),
        sft_epochs=grpo.get("sft_epochs", 2),
        sft_lr=grpo.get("sft_lr", 1e-4),
        domain=env.get("domain", "banking"),
        intents=env.get("intents", ["transfer", "check_balance", "block_card"]),
        output_dir=paths.get("output_dir", "./grpo_output"),
    )


def make_env_config(cfg: dict[str, Any]) -> EnvConfig:
    """Build an EnvConfig from the loaded YAML dict."""
    env = cfg.get("environment", {})
    reward = cfg.get("reward", {})

    reward_config = RewardConfig(
        intent_correct_bonus=reward.get("intent_correct_bonus", 50.0),
        intent_wrong_penalty=reward.get("intent_wrong_penalty", -50.0),
        fast_bonus=reward.get("fast_bonus", 20.0),
        medium_bonus=reward.get("medium_bonus", 10.0),
        slow_penalty_per_turn=reward.get("slow_penalty_per_turn", -5.0),
        injection_caught_bonus=reward.get("injection_caught_bonus", 40.0),
        injection_succeeded_penalty=reward.get("injection_succeeded_penalty", -100.0),
        api_correct_bonus=reward.get("api_correct_bonus", 20.0),
        api_wrong_penalty=reward.get("api_wrong_penalty", -30.0),
        helpfulness_bonus=reward.get("helpfulness_bonus", 15.0),
        prompt_length_threshold=reward.get("prompt_length_threshold", 300),
        prompt_length_penalty_per_token=reward.get("prompt_length_penalty_per_token", -0.1),
        no_intent_penalty=reward.get("no_intent_penalty", -20.0),
    )

    return EnvConfig(
        domain=env.get("domain", "banking"),
        intents=env.get("intents", ["transfer", "check_balance", "block_card"]),
        max_turns=env.get("max_turns", 10),
        reward_config=reward_config,
    )


def get_report_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract report settings from config."""
    report = cfg.get("report", {})
    return {
        "enabled": report.get("enabled", True),
        "output_dir": report.get("output_dir", "./reports"),
        "eval_episodes": report.get("eval_episodes", 5),
        "example_customers": report.get("example_customers", 3),
    }


def get_paths(cfg: dict[str, Any]) -> dict[str, str]:
    """Extract path settings from config."""
    paths = cfg.get("paths", {})
    return {
        "output_dir": paths.get("output_dir", "./grpo_output"),
        "log_dir": paths.get("log_dir", "./logs"),
    }


def get_generation_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract generation/inference settings from config."""
    gen = cfg.get("generation", {})
    return {
        "inference_backend": gen.get("inference_backend", "auto"),
        "max_seq_length": gen.get("max_seq_length", 4096),
        "prompt_max_new_tokens": gen.get("prompt_max_new_tokens", 2048),
        "prompt_temperature": gen.get("prompt_temperature", 0.3),
        "agent_max_tokens": gen.get("agent_max_tokens", 300),
        "agent_temperature": gen.get("agent_temperature", 0.3),
        "customer_max_tokens": gen.get("customer_max_tokens", 200),
        "customer_temperature": gen.get("customer_temperature", 0.7),
    }


def get_upload_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract Supabase upload settings from config."""
    upload = cfg.get("upload", {})
    return {
        "enabled": upload.get("enabled", False),
        "bucket": upload.get("bucket", "training-results"),
    }


def get_personas_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract persona settings from config."""
    personas = cfg.get("personas", {})
    return {
        "count": personas.get("count", 100),
    }
