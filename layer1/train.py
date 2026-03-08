"""
Layer 1 — Executable GRPO training script.

Usage:
    # Full GPU training (requires Colab/GPU + train deps)
    python -m layer1.train --mode train --steps 50

    # CPU mock optimization (evaluates hand-written prompts)
    python -m layer1.train --mode mock --episodes 20

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

from layer1.grpo_trainer import (
    GRPOConfig,
    GRPOPromptTrainer,
    MockPromptOptimizer,
    PromptEvaluator,
    build_meta_prompt,
)
from layer1.training_logger import TrainingLogger, ReportGenerator
from layer2.customer_sim import CustomerPersona, CustomerSimulator
from layer2.hf_agent import HFAgent
from personas.generate_personas import generate_personas

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def load_evaluator(hf_token: str | None = None, use_llm_agent: bool = False) -> PromptEvaluator:
    """Load personas and create the evaluator with optional LLM agent."""
    token = hf_token or os.environ.get("HF_TOKEN")
    personas_data = generate_personas(100)
    personas = [CustomerPersona(**p) for p in personas_data]
    simulator = CustomerSimulator(hf_token=token)

    agent_fn = None
    if use_llm_agent and token:
        agent = HFAgent(hf_token=token)
        if agent.is_llm_available:
            agent_fn = agent
            logger.info("Using LLM agent (Llama 3.1 8B)")
        else:
            logger.warning("LLM agent not available, using rule-based fallback")

    return PromptEvaluator(personas=personas, simulator=simulator, agent_fn=agent_fn)


def run_mock(args):
    """Run mock optimization with hand-written prompts."""
    evaluator = load_evaluator(args.hf_token, use_llm_agent=args.llm_agent)
    training_logger = TrainingLogger(
        log_dir=args.log_dir,
        total_steps=len(MockPromptOptimizer.CANDIDATE_PROMPTS),
    )
    optimizer = MockPromptOptimizer(evaluator, logger=training_logger)
    result = optimizer.optimize(num_episodes_per_prompt=args.episodes)

    print(f"\n{'='*60}")
    print("MOCK OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    for r in optimizer.results:
        print(f"  Prompt {r['prompt_index']}: reward={r['mean_reward']:.1f}")
    print(f"\nBest prompt (reward={result['best_reward']:.1f}):")
    print(result["best_prompt"])

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    if args.report:
        print(f"\n{'='*60}")
        print("GENERATING TRAINING REPORT...")
        print(f"{'='*60}")
        report_gen = ReportGenerator(evaluator, training_logger)
        report_path = report_gen.generate_report(
            output_dir=args.report_dir,
            num_eval_episodes=args.eval_episodes,
            num_example_customers=args.example_customers,
        )
        print(f"\nReport saved to {report_path}")


def run_train(args):
    """Run full GRPO training (requires GPU)."""
    evaluator = load_evaluator(args.hf_token, use_llm_agent=args.llm_agent)
    training_logger = TrainingLogger(log_dir=args.log_dir, total_steps=args.steps)
    config = GRPOConfig(
        num_training_steps=args.steps,
        episodes_per_candidate=args.episodes,
        output_dir=args.output_dir,
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
    result = evaluator.evaluate_prompt(best_prompt, num_episodes=args.episodes)
    print(f"\nEvaluation: mean_reward={result['mean_reward']:.1f}")

    if args.report:
        print(f"\n{'='*60}")
        print("GENERATING TRAINING REPORT...")
        print(f"{'='*60}")
        report_gen = ReportGenerator(evaluator, training_logger)
        report_path = report_gen.generate_report(
            output_dir=args.report_dir,
            num_eval_episodes=args.eval_episodes,
            num_example_customers=args.example_customers,
        )
        print(f"\nReport saved to {report_path}")


def run_eval(args):
    """Evaluate a single prompt."""
    evaluator = load_evaluator(args.hf_token, use_llm_agent=args.llm_agent)
    result = evaluator.evaluate_prompt(args.prompt, num_episodes=args.episodes)
    print(f"Prompt: {args.prompt[:80]}...")
    print(f"Mean reward: {result['mean_reward']:.1f}")
    print(f"Min/Max: {result['min_reward']:.1f} / {result['max_reward']:.1f}")

    # Show per-episode breakdown
    for i, log in enumerate(result["logs"]):
        print(
            f"  Episode {i}: intent={log['true_intent']} "
            f"correct={log['intent_correct']} turns={log['turns']} "
            f"reward={result['rewards'][i]:.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Layer 1 — GRPO Prompt Optimizer")
    parser.add_argument(
        "--mode",
        choices=["train", "mock", "eval"],
        default="mock",
        help="Training mode: train (GPU), mock (CPU), eval (single prompt)",
    )
    parser.add_argument("--episodes", type=int, default=7, help="Episodes per evaluation")
    parser.add_argument("--steps", type=int, default=10, help="GRPO training steps (train mode)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--output-dir", type=str, default="./grpo_output", help="Training output dir")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to evaluate (eval mode)")
    parser.add_argument("--llm-agent", action="store_true",
                        help="Use LLM (Llama 3.1) as the agent instead of rule-based")
    parser.add_argument("--report", action="store_true", default=True,
                        help="Generate training report after completion (default: True)")
    parser.add_argument("--no-report", action="store_false", dest="report",
                        help="Skip report generation")
    parser.add_argument("--report-dir", type=str, default="./reports",
                        help="Directory for report output")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for training logs")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Episodes per checkpoint for report evaluation")
    parser.add_argument("--example-customers", type=int, default=3,
                        help="Number of example customers in report")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "mock":
        run_mock(args)
    elif args.mode == "eval":
        if not args.prompt:
            parser.error("--prompt is required for eval mode")
        run_eval(args)


if __name__ == "__main__":
    main()
