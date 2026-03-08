"""
Layer 1 — GRPO training script for prompt optimization.

All parameters are loaded from config.yaml (single source of truth).
CLI flags override config.yaml values.

Usage:
    # Train with defaults from config.yaml
    python -m layer1.train

    # Override specific params
    python -m layer1.train --steps 20 --episodes 10

    # Evaluate a single prompt
    python -m layer1.train --mode eval --prompt "You are a helpful agent."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os

# Auto-load .env for HF_TOKEN
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config, make_grpo_config, make_env_config, get_report_config, get_paths
from layer1.grpo_trainer import GRPOConfig, GRPOPromptTrainer, PromptEvaluator
from layer1.training_logger import TrainingLogger, ReportGenerator
from layer2.customer_sim import CustomerPersona, CustomerSimulator
from layer2.hf_agent import HFAgent
from personas.generate_personas import generate_personas

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def load_evaluator(hf_token: str | None = None) -> PromptEvaluator:
    """Load personas and create the evaluator with LLM agent."""
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required. Set it via --hf-token or the HF_TOKEN environment variable."
        )

    personas_data = generate_personas(100)
    personas = [CustomerPersona(**p) for p in personas_data]
    simulator = CustomerSimulator(hf_token=token)

    agent = HFAgent(hf_token=token)
    if not agent.is_llm_available:
        raise RuntimeError(
            "LLM agent could not be initialized. Check your HF_TOKEN and huggingface_hub installation."
        )
    logger.info("Using LLM agent (Llama 3.1 8B)")

    return PromptEvaluator(personas=personas, simulator=simulator, agent_fn=agent)


def _print_config_banner(config: GRPOConfig, report_cfg: dict, paths_cfg: dict):
    """Print all training parameters from config."""
    total_conversations = (
        config.num_training_steps * config.num_candidates * config.episodes_per_candidate
    )

    print(f"\n{'='*70}")
    print(f"  TRAINING CONFIGURATION (from config.yaml)")
    print(f"{'='*70}")
    print()
    print(f"  --- Layer 1: GRPO RL Training ---")
    print(f"  Prompt Generator Model:        {config.model_name}")
    print(f"  LoRA:                          r={config.lora_r}  alpha={config.lora_alpha}  dropout={config.lora_dropout}")
    print(f"  Learning Rate:                 {config.learning_rate:.1e}")
    print(f"  Steps / GRPO Iterations:       {config.num_training_steps}")
    print(f"  Candidates / Customer Reps:    {config.num_candidates} per step")
    print(f"  Episodes / Customers:          {config.episodes_per_candidate} per candidate")
    print(f"  Max Prompt Length:             {config.max_prompt_length} tokens")
    print(f"  Batch Size:                    {config.per_device_train_batch_size}")
    print(f"  Gradient Accumulation:         {config.gradient_accumulation_steps}")
    print()
    print(f"  --- Layer 2: Conversation Environment ---")
    print(f"  Domain:                        {config.domain}")
    print(f"  Intents:                       {config.intents}")
    print(f"  Max Turns per Conversation:    (from env config)")
    print(f"  Customer Rep Agent:            Llama 3.1 8B (HF Inference API)")
    print(f"  Customer Simulator:            Llama 3.1 8B (HF Inference API)")
    print()
    print(f"  --- Totals ---")
    print(f"  Total LLM Conversations:       ~{total_conversations}")
    print(f"  Report Generation:             {'yes' if report_cfg['enabled'] else 'no'}")
    print(f"  Output Dir:                    {paths_cfg['output_dir']}")
    print(f"  Log Dir:                       {paths_cfg['log_dir']}")
    print(f"{'='*70}\n")


def run_train(config: GRPOConfig, report_cfg: dict, paths_cfg: dict, hf_token: str | None):
    """Run GRPO training."""
    _print_config_banner(config, report_cfg, paths_cfg)
    evaluator = load_evaluator(hf_token)
    training_logger = TrainingLogger(
        log_dir=paths_cfg["log_dir"], total_steps=config.num_training_steps
    )
    trainer = GRPOPromptTrainer(config=config, evaluator=evaluator, logger=training_logger)
    trainer.setup_model()
    trainer.train()

    best_prompt = trainer.generate_best_prompt()
    print(f"\n{'='*60}")
    print("TRAINED SYSTEM PROMPT")
    print(f"{'='*60}")
    print(best_prompt)

    # Evaluate the trained prompt
    result = evaluator.evaluate_prompt(
        best_prompt, num_episodes=config.episodes_per_candidate
    )
    print(f"\nEvaluation: mean_reward={result['mean_reward']:.1f}")

    if report_cfg["enabled"]:
        print(f"\n{'='*60}")
        print("GENERATING TRAINING REPORT...")
        print(f"{'='*60}")
        report_gen = ReportGenerator(evaluator, training_logger)
        report_path = report_gen.generate_report(
            output_dir=report_cfg["output_dir"],
            num_eval_episodes=report_cfg["eval_episodes"],
            num_example_customers=report_cfg["example_customers"],
        )
        print(f"\nReport saved to {report_path}")


def run_eval(hf_token: str | None, prompt: str, episodes: int):
    """Evaluate a single prompt."""
    evaluator = load_evaluator(hf_token)
    result = evaluator.evaluate_prompt(prompt, num_episodes=episodes)
    print(f"Prompt: {prompt[:80]}...")
    print(f"Mean reward: {result['mean_reward']:.1f}")
    print(f"Min/Max: {result['min_reward']:.1f} / {result['max_reward']:.1f}")

    for i, log in enumerate(result["logs"]):
        print(
            f"  Episode {i}: intent={log['true_intent']} "
            f"correct={log['intent_correct']} turns={log['turns']} "
            f"reward={result['rewards'][i]:.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Layer 1 — GRPO Prompt Optimizer")
    parser.add_argument(
        "--mode", choices=["train", "eval"], default="train",
        help="Mode: train (GRPO RL training), eval (evaluate a single prompt)",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episodes_per_candidate from config")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override num_training_steps from config")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace API token")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to evaluate (eval mode)")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip report generation")
    parser.add_argument("--report-dir", type=str, default=None,
                        help="Override report output directory from config")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Override log directory from config")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="Override eval episodes for report from config")
    parser.add_argument("--example-customers", type=int, default=None,
                        help="Override example customers in report from config")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    # Legacy flags (accepted for backwards compatibility with external runners)
    parser.add_argument("--llm-agent", action="store_true", default=True,
                        help="(deprecated, always true) Use LLM agent")
    args = parser.parse_args()

    # Load config from YAML
    cfg = load_config(args.config)
    grpo_config = make_grpo_config(cfg)
    report_cfg = get_report_config(cfg)
    paths_cfg = get_paths(cfg)

    # CLI overrides
    if args.steps is not None:
        grpo_config.num_training_steps = args.steps
    if args.episodes is not None:
        grpo_config.episodes_per_candidate = args.episodes
    if args.output_dir is not None:
        grpo_config.output_dir = args.output_dir
        paths_cfg["output_dir"] = args.output_dir
    if args.no_report:
        report_cfg["enabled"] = False
    if args.report_dir is not None:
        report_cfg["output_dir"] = args.report_dir
    if args.log_dir is not None:
        paths_cfg["log_dir"] = args.log_dir
    if args.eval_episodes is not None:
        report_cfg["eval_episodes"] = args.eval_episodes
    if args.example_customers is not None:
        report_cfg["example_customers"] = args.example_customers

    if args.mode == "train":
        run_train(grpo_config, report_cfg, paths_cfg, args.hf_token)
    elif args.mode == "eval":
        if not args.prompt:
            parser.error("--prompt is required for eval mode")
        episodes = args.episodes or grpo_config.episodes_per_candidate
        run_eval(args.hf_token, args.prompt, episodes)


if __name__ == "__main__":
    main()
