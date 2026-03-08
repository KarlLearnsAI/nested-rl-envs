"""
Training Logger & Report Generator for the RL Prompt Optimization pipeline.

TrainingLogger: Appends per-iteration logs to a text file and keeps structured data in memory.
ReportGenerator: After training, evaluates checkpoint prompts and produces a markdown report
                 with reward charts and side-by-side conversation comparisons.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from typing import Any

from layer0.reward import reward_fn
from layer2.customer_sim import CustomerPersona

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Logs each training iteration to a text file and stores structured data."""

    def __init__(self, log_dir: str = "./logs", total_steps: int | None = None):
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"log_{self.timestamp}.txt")
        self.json_path = os.path.join(log_dir, f"training_{self.timestamp}.json")
        self.total_steps = total_steps
        self.iterations: list[dict[str, Any]] = []
        self._start_time = datetime.now()

        with open(self.log_path, "w") as f:
            f.write(f"Training Log — {self._start_time.isoformat()}\n")
            f.write(f"{'=' * 60}\n\n")
            f.flush()
            os.fsync(f.fileno())

    def log_iteration(self, step: int, prompt: str, eval_result: dict[str, Any]):
        """Log a single training iteration (one prompt evaluated)."""
        entry = {
            "step": step,
            "prompt": prompt,
            "mean_reward": eval_result.get("mean_reward", 0.0),
            "min_reward": eval_result.get("min_reward", 0.0),
            "max_reward": eval_result.get("max_reward", 0.0),
            "num_episodes": eval_result.get("num_episodes", 0),
            "rewards": eval_result.get("rewards", []),
            "logs": eval_result.get("logs", []),
        }
        self.iterations.append(entry)

        total_str = f" / {self.total_steps}" if self.total_steps else ""
        with open(self.log_path, "a") as f:
            f.write(f"=== Step {step}{total_str} ===\n")
            f.write(f"Prompt: {prompt[:300]}{'...' if len(prompt) > 300 else ''}\n")
            f.write(f"Mean Reward: {entry['mean_reward']:.1f}\n")
            f.write(f"Min/Max: {entry['min_reward']:.1f} / {entry['max_reward']:.1f}\n")
            f.write(f"Episodes: {entry['num_episodes']}\n")
            f.write(f"---\n\n")
            f.flush()
            os.fsync(f.fileno())

        # Incremental save — persist JSON after every step so data survives crashes
        self.save_json()

        logger.info("Logged step %d: mean_reward=%.1f", step, entry["mean_reward"])

    def save_json(self):
        """Save structured training data to JSON."""
        data = {
            "timestamp": self.timestamp,
            "start_time": self._start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_steps": self.total_steps,
            "iterations": [
                {k: v for k, v in it.items() if k != "logs"}
                for it in self.iterations
            ],
        }
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        logger.info("Training data saved to %s", self.json_path)

    def get_checkpoint_indices(self) -> list[int]:
        """Return indices for 3 checkpoints: start, middle, end."""
        n = len(self.iterations)
        if n <= 1:
            return list(range(n))
        if n == 2:
            return [0, n - 1]
        return [0, n // 2, n - 1]


def _select_diverse_personas(
    personas: list[CustomerPersona], count: int = 10
) -> list[CustomerPersona]:
    """Select a diverse set of personas covering intents, personalities, and SE types."""
    buckets: dict[str, list[CustomerPersona]] = {}
    for p in personas:
        key = f"{p.true_intent}_{p.social_engineering}"
        buckets.setdefault(key, []).append(p)

    selected: list[CustomerPersona] = []
    # Round-robin across buckets to ensure diversity
    bucket_iters = {k: iter(v) for k, v in buckets.items()}
    while len(selected) < count and bucket_iters:
        empty_keys = []
        for key, it in bucket_iters.items():
            if len(selected) >= count:
                break
            try:
                selected.append(next(it))
            except StopIteration:
                empty_keys.append(key)
        for k in empty_keys:
            del bucket_iters[k]

    # If we still need more, fill randomly
    if len(selected) < count:
        remaining = [p for p in personas if p not in selected]
        selected.extend(random.sample(remaining, min(count - len(selected), len(remaining))))

    return selected


class ReportGenerator:
    """Generates a training report with charts and conversation comparisons."""

    def __init__(self, evaluator: Any, training_logger: TrainingLogger):
        self.evaluator = evaluator
        self.logger = training_logger

    def generate_report(
        self,
        output_dir: str = "./reports",
        num_eval_episodes: int = 30,
        num_example_customers: int = 10,
    ) -> str:
        """Generate the full training report. Returns path to the markdown report."""
        os.makedirs(output_dir, exist_ok=True)
        ts = self.logger.timestamp

        # 1. Select checkpoint prompts
        indices = self.logger.get_checkpoint_indices()
        checkpoints = [self.logger.iterations[i] for i in indices]
        checkpoint_labels = self._make_labels(indices)

        # 2. Evaluate each checkpoint on same personas for fair comparison
        eval_personas = random.sample(
            self.evaluator.env.personas,
            min(num_eval_episodes, len(self.evaluator.env.personas)),
        )
        checkpoint_evals = []
        for i, cp in enumerate(checkpoints):
            logger.info(
                "Evaluating checkpoint %s (%d/%d)...",
                checkpoint_labels[i], i + 1, len(checkpoints),
            )
            result = self.evaluator.evaluate_prompt(
                cp["prompt"],
                num_episodes=num_eval_episodes,
                personas_subset=eval_personas,
            )
            result["label"] = checkpoint_labels[i]
            result["step"] = cp["step"]
            result["prompt"] = cp["prompt"]
            result["training_mean_reward"] = cp["mean_reward"]
            checkpoint_evals.append(result)

        # 3. Generate reward chart
        chart_path = os.path.join(output_dir, f"reward_chart_{ts}.png")
        self._generate_chart(checkpoint_evals, chart_path)

        # 4. Run example conversations
        example_personas = _select_diverse_personas(
            self.evaluator.env.personas, num_example_customers
        )
        example_conversations = self._run_example_conversations(
            checkpoints, checkpoint_labels, example_personas
        )

        # 5. Compute per-checkpoint metrics
        checkpoint_metrics = self._compute_metrics(checkpoint_evals)

        # 6. Write markdown report
        report_path = os.path.join(output_dir, f"report_{ts}.md")
        self._write_report(
            report_path, chart_path, checkpoints, checkpoint_labels,
            checkpoint_evals, checkpoint_metrics, example_conversations,
        )

        # 7. Save logger JSON
        self.logger.save_json()

        logger.info("Report generated: %s", report_path)
        return report_path

    def _make_labels(self, indices: list[int]) -> list[str]:
        labels = []
        for i, idx in enumerate(indices):
            step = self.logger.iterations[idx]["step"]
            if i == 0:
                labels.append(f"Step {step} (Start)")
            elif i == len(indices) - 1:
                labels.append(f"Step {step} (Final)")
            else:
                labels.append(f"Step {step} (Mid)")
        return labels

    def _generate_chart(self, checkpoint_evals: list[dict], save_path: str):
        """Generate reward progression chart using matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Line chart of all training iterations
        ax1 = axes[0]
        steps = [it["step"] for it in self.logger.iterations]
        rewards = [it["mean_reward"] for it in self.logger.iterations]
        ax1.plot(steps, rewards, "b-o", markersize=4, alpha=0.7, label="Training reward")

        # Highlight checkpoints
        cp_steps = [e["step"] for e in checkpoint_evals]
        cp_rewards_training = [e["training_mean_reward"] for e in checkpoint_evals]
        ax1.scatter(cp_steps, cp_rewards_training, c="red", s=100, zorder=5, label="Checkpoints")
        for e in checkpoint_evals:
            ax1.annotate(
                e["label"].split("(")[1].rstrip(")"),
                (e["step"], e["training_mean_reward"]),
                textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9,
            )

        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Mean Reward")
        ax1.set_title("Reward Progression During Training")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Right: Grouped bar chart of checkpoint evaluation metrics
        ax2 = axes[1]
        labels = [e["label"] for e in checkpoint_evals]
        eval_rewards = [e["mean_reward"] for e in checkpoint_evals]
        x = range(len(labels))
        bars = ax2.bar(x, eval_rewards, color=["#e74c3c", "#f39c12", "#27ae60"])
        ax2.set_xticks(list(x))
        ax2.set_xticklabels([l.split("(")[1].rstrip(")") for l in labels])
        ax2.set_ylabel(f"Mean Reward ({checkpoint_evals[0].get('num_episodes', '?')}-episode eval)")
        ax2.set_title("Checkpoint Comparison")
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, eval_rewards):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Chart saved to %s", save_path)

    def _run_example_conversations(
        self,
        checkpoints: list[dict],
        checkpoint_labels: list[str],
        example_personas: list[CustomerPersona],
    ) -> list[dict]:
        """Run each example persona through all checkpoint agents."""
        examples = []
        for pi, persona in enumerate(example_personas):
            customer_data = {
                "persona_id": persona.id,
                "true_intent": persona.true_intent,
                "personality": persona.personality,
                "social_engineering": persona.social_engineering,
                "conversations": [],
            }
            for ci, cp in enumerate(checkpoints):
                log = self.evaluator.env.run_episode(
                    system_prompt=cp["prompt"],
                    agent_fn=self.evaluator.agent_fn,
                    persona=persona,
                )
                r = reward_fn(log)
                customer_data["conversations"].append({
                    "label": checkpoint_labels[ci],
                    "reward": r,
                    "turns": log.turns,
                    "intent_correct": log.intent_correct,
                    "injection_attempted": log.injection_attempted,
                    "injection_succeeded": log.injection_succeeded,
                    "messages": log.messages,
                })
            examples.append(customer_data)
            logger.info("Example %d/%d complete", pi + 1, len(example_personas))
        return examples

    def _compute_metrics(self, checkpoint_evals: list[dict]) -> list[dict]:
        """Compute aggregate metrics for each checkpoint evaluation."""
        metrics = []
        for e in checkpoint_evals:
            logs = e.get("logs", [])
            total = len(logs)
            if total == 0:
                metrics.append({
                    "label": e["label"],
                    "mean_reward": 0,
                    "intent_accuracy": 0,
                    "injection_resistance": "N/A",
                    "avg_turns": 0,
                })
                continue

            correct = sum(1 for l in logs if l.get("intent_correct"))
            inj_attempted = sum(1 for l in logs if l.get("injection_attempted"))
            inj_succeeded = sum(1 for l in logs if l.get("injection_succeeded"))
            avg_turns = sum(l.get("turns", 0) for l in logs) / total

            metrics.append({
                "label": e["label"],
                "mean_reward": e["mean_reward"],
                "intent_accuracy": f"{correct / total:.0%}",
                "injection_resistance": (
                    f"{(inj_attempted - inj_succeeded) / inj_attempted:.0%}"
                    if inj_attempted > 0
                    else "N/A (none attempted)"
                ),
                "avg_turns": f"{avg_turns:.1f}",
            })
        return metrics

    def _write_report(
        self,
        report_path: str,
        chart_path: str,
        checkpoints: list[dict],
        checkpoint_labels: list[str],
        checkpoint_evals: list[dict],
        checkpoint_metrics: list[dict],
        example_conversations: list[dict],
    ):
        """Write the final markdown report."""
        duration = datetime.now() - self.logger._start_time
        chart_filename = os.path.basename(chart_path)

        lines = []
        lines.append(f"# Training Report — {self.logger._start_time.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("## Training Summary")
        lines.append(f"- **Total steps:** {len(self.logger.iterations)}")
        lines.append(f"- **Duration:** {duration.seconds // 60}m {duration.seconds % 60}s")
        lines.append(f"- **Checkpoints evaluated on:** {checkpoint_evals[0].get('num_episodes', 30)} episodes each")
        lines.append("")

        # Checkpoint prompts
        lines.append("## Checkpoint Prompts")
        lines.append("")
        for i, cp in enumerate(checkpoints):
            mr = cp["mean_reward"]
            lines.append(f"### {checkpoint_labels[i]} (Training Reward: {mr:.1f})")
            lines.append("")
            lines.append("```")
            lines.append(cp["prompt"])
            lines.append("```")
            lines.append("")

        # Reward chart
        lines.append("## Reward Progression")
        lines.append("")
        lines.append(f"![Reward Chart]({chart_filename})")
        lines.append("")

        # Metrics table
        lines.append("## Checkpoint Comparison")
        lines.append("")
        lines.append("| Checkpoint | Mean Reward | Intent Accuracy | Injection Resistance | Avg Turns |")
        lines.append("|------------|-------------|-----------------|----------------------|-----------|")
        for m in checkpoint_metrics:
            lines.append(
                f"| {m['label']} | {m['mean_reward']:.1f} | {m['intent_accuracy']} "
                f"| {m['injection_resistance']} | {m['avg_turns']} |"
            )
        lines.append("")

        # Example conversations
        lines.append(f"## Example Conversations ({len(example_conversations)} Customers x {len(checkpoints)} Agents)")
        lines.append("")
        customer_letters = "ABCDEFGHIJ"
        for ci, cust in enumerate(example_conversations):
            letter = customer_letters[ci] if ci < len(customer_letters) else str(ci)
            lines.append(
                f"### Customer {letter} — Persona {cust['persona_id']} "
                f"({cust['true_intent']}, {cust['personality']}, "
                f"SE={cust['social_engineering']})"
            )
            lines.append("")

            for conv in cust["conversations"]:
                status = ""
                if conv["injection_attempted"]:
                    status = " | INJECTION " + ("BLOCKED" if not conv["injection_succeeded"] else "SUCCEEDED")
                lines.append(
                    f"#### {conv['label']} (Reward: {conv['reward']:.1f}, "
                    f"Turns: {conv['turns']}, "
                    f"Intent: {'correct' if conv['intent_correct'] else 'wrong'}"
                    f"{status})"
                )
                lines.append("")
                for msg in conv["messages"]:
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown").capitalize()
                        content = msg.get("content", "")
                        lines.append(f"**{role}:** {content}")
                        lines.append("")
                lines.append("---")
                lines.append("")

            lines.append("")

        with open(report_path, "w") as f:
            f.write("\n".join(lines))
            f.flush()
            os.fsync(f.fileno())
