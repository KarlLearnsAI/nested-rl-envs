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
from datetime import datetime

# Auto-load .env for HF_TOKEN
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config, make_grpo_config, make_env_config, get_report_config, get_paths, get_generation_config, get_personas_config, get_upload_config
from layer1.grpo_trainer import GRPOConfig, GRPOPromptTrainer, PromptEvaluator
from layer1.training_logger import TrainingLogger, ReportGenerator
from layer1.upload import SupabaseUploader
from layer2.customer_sim import CustomerPersona, CustomerSimulator
from layer2.hf_agent import HFAgent
from personas.generate_personas import generate_personas

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def verify_volume_mount(paths_cfg: dict) -> None:
    """Write a canary file at startup to verify the volume is mounted and writable."""
    output_dirs = [
        paths_cfg.get("output_dir", ""),
        paths_cfg.get("log_dir", ""),
    ]
    for d in output_dirs:
        if not d:
            continue
        os.makedirs(d, exist_ok=True)
        canary = os.path.join(d, ".volume_check")
        try:
            with open(canary, "w") as f:
                f.write(f"volume check {datetime.now().isoformat()}\n")
                f.flush()
                os.fsync(f.fileno())
            logger.info("Volume check OK: %s", d)
        except OSError as e:
            logger.error("VOLUME WRITE FAILED for %s: %s", d, e)
            print(f"\n*** WARNING: Cannot write to {d} — volume may not be mounted! ***\n")
            raise


def _try_load_local_model(gen_cfg: dict, hf_token: str | None):
    """Try to load Llama locally; returns LocalLlamaModel or None."""
    backend = gen_cfg.get("inference_backend", "auto")
    if backend == "api":
        logger.info("inference_backend=api — using HF Inference API")
        return None

    try:
        import torch
        if not torch.cuda.is_available():
            if backend == "local":
                raise RuntimeError("inference_backend=local but no CUDA GPU available")
            logger.info("No CUDA GPU — falling back to HF Inference API")
            return None

        from layer2.local_model import get_shared_model
        model = get_shared_model(hf_token=hf_token)
        logger.info("Using local Llama model for Layer 2 inference")
        return model
    except ImportError:
        if backend == "local":
            raise RuntimeError(
                "inference_backend=local but torch/transformers not installed. "
                "Install with: pip install -e '.[train]'"
            )
        logger.info("torch/transformers not available — using HF Inference API")
        return None


def load_evaluator(
    hf_token: str | None = None,
    gen_cfg: dict | None = None,
    personas_cfg: dict | None = None,
) -> PromptEvaluator:
    """Load personas and create the evaluator with LLM agent."""
    token = hf_token or os.environ.get("HF_TOKEN")
    gen_cfg = gen_cfg or {}
    personas_cfg = personas_cfg or {}

    # Try local model first
    local_model = _try_load_local_model(gen_cfg, token)

    if local_model is None and not token:
        raise RuntimeError(
            "HF_TOKEN is required when not using local model. "
            "Set it via --hf-token or the HF_TOKEN environment variable."
        )

    persona_count = personas_cfg.get("count", 100)
    personas_data = generate_personas(persona_count)
    personas = [CustomerPersona(**p) for p in personas_data]
    simulator = CustomerSimulator(
        hf_token=token,
        max_tokens=gen_cfg.get("customer_max_tokens", 200),
        temperature=gen_cfg.get("customer_temperature", 0.7),
        local_model=local_model,
    )

    agent = HFAgent(
        hf_token=token,
        max_tokens=gen_cfg.get("agent_max_tokens", 300),
        temperature=gen_cfg.get("agent_temperature", 0.3),
        local_model=local_model,
    )
    if not agent.is_llm_available:
        raise RuntimeError(
            "LLM agent could not be initialized. Check your HF_TOKEN and huggingface_hub installation."
        )

    backend_name = "local model" if local_model else "HF Inference API"
    logger.info("Using LLM agent (Llama 3.1 8B via %s)", backend_name)

    return PromptEvaluator(personas=personas, simulator=simulator, agent_fn=agent)


def _print_config_banner(config: GRPOConfig, report_cfg: dict, paths_cfg: dict, gen_cfg: dict | None = None):
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
    print(f"  Inference Backend:             {gen_cfg.get('inference_backend', 'auto') if gen_cfg else 'auto'}")
    print(f"  Customer Rep Agent:            Llama 3.1 8B")
    print(f"  Customer Simulator:            Llama 3.1 8B")
    print()
    print(f"  --- Totals ---")
    print(f"  Total LLM Conversations:       ~{total_conversations}")
    print(f"  Report Generation:             {'yes' if report_cfg['enabled'] else 'no'}")
    print(f"  Output Dir:                    {paths_cfg['output_dir']}")
    print(f"  Log Dir:                       {paths_cfg['log_dir']}")
    print(f"{'='*70}\n")


def run_train(config: GRPOConfig, report_cfg: dict, paths_cfg: dict, hf_token: str | None, gen_cfg: dict | None = None, personas_cfg: dict | None = None, upload_cfg: dict | None = None):
    """Run GRPO training."""
    _print_config_banner(config, report_cfg, paths_cfg, gen_cfg=gen_cfg)

    # Verify volume is mounted before doing any expensive work
    all_paths = dict(paths_cfg)
    if report_cfg.get("enabled") and report_cfg.get("output_dir"):
        all_paths["report_dir"] = report_cfg["output_dir"]
    verify_volume_mount(all_paths)

    evaluator = load_evaluator(hf_token, gen_cfg=gen_cfg, personas_cfg=personas_cfg)
    training_logger = TrainingLogger(
        log_dir=paths_cfg["log_dir"], total_steps=config.num_training_steps
    )

    # Wire up incremental Supabase uploads
    upload_cfg = upload_cfg or {}
    uploader = None
    if upload_cfg.get("enabled") and os.environ.get("SUPABASE_URL"):
        uploader = SupabaseUploader(
            run_id=training_logger.timestamp,
            bucket=upload_cfg.get("bucket", "training-results"),
            config={"grpo": config.__dict__, "report": report_cfg, "paths": paths_cfg},
        )
        if uploader.enabled:
            training_logger.add_on_step_callback(uploader.after_step)
            print("Supabase incremental upload enabled")
        else:
            uploader = None
    elif upload_cfg.get("enabled"):
        print("Supabase upload enabled but SUPABASE_URL not set — skipping")

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

    # Always output raw training summary (arrays for post-hoc analysis)
    print(f"\n{'='*60}")
    print("RAW TRAINING SUMMARY")
    print(f"{'='*60}")
    raw_summary = training_logger.generate_raw_summary()
    summary_path = training_logger.save_raw_summary(paths_cfg.get("log_dir"))
    print(f"Saved to: {summary_path}")
    print(f"\nSteps:        {raw_summary['steps']}")
    print(f"Mean rewards: {raw_summary['mean_rewards']}")
    print(f"Min rewards:  {raw_summary['min_rewards']}")
    print(f"Max rewards:  {raw_summary['max_rewards']}")
    print(f"Best step:    {raw_summary['best_step']} (reward={raw_summary['best_mean_reward']})")
    print(f"Total episodes: {raw_summary['total_episodes']}")
    print(f"Duration:     {raw_summary['duration_seconds']}s")
    print(f"\nPer-step episode rewards:")
    for step, rewards in zip(raw_summary["steps"], raw_summary["all_episode_rewards"]):
        print(f"  Step {step:3d}: {rewards}")
    print(f"\nFull raw JSON: {summary_path}")
    print(f"{'='*60}")

    report_path = None
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

        # Print report to stdout as fallback (always visible in logs)
        try:
            with open(report_path, "r") as f:
                report_content = f.read()
            print(f"\n{'='*60}")
            print("REPORT CONTENT (stdout fallback)")
            print(f"{'='*60}")
            print(report_content)
            print(f"{'='*60}")
        except OSError:
            print("WARNING: Could not re-read report from disk")

    # Finalize Supabase upload (update duration, upload files)
    if uploader and uploader.enabled:
        print(f"\n{'='*60}")
        print("FINALIZING SUPABASE UPLOAD...")
        print(f"{'='*60}")
        uploader.finish(
            duration_seconds=raw_summary.get("duration_seconds"),
            report_path=report_path,
            raw_summary=raw_summary,
        )
        print(f"  Run ID:  {uploader.run_id}")
        print(f"  Steps uploaded incrementally: {len(uploader._mean_rewards)}")
        print(f"  Episodes uploaded: {uploader._total_episodes}")
        print(f"{'='*60}")


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
    args = parser.parse_args()

    # Load config from YAML
    cfg = load_config(args.config)
    grpo_config = make_grpo_config(cfg)
    report_cfg = get_report_config(cfg)
    paths_cfg = get_paths(cfg)
    gen_cfg = get_generation_config(cfg)
    personas_cfg = get_personas_config(cfg)
    upload_cfg = get_upload_config(cfg)

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
        run_train(grpo_config, report_cfg, paths_cfg, args.hf_token, gen_cfg=gen_cfg, personas_cfg=personas_cfg, upload_cfg=upload_cfg)
    elif args.mode == "eval":
        if not args.prompt:
            parser.error("--prompt is required for eval mode")
        episodes = args.episodes or grpo_config.episodes_per_candidate
        run_eval(args.hf_token, args.prompt, episodes)


if __name__ == "__main__":
    main()
