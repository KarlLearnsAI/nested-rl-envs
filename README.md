---
title: Nested RL Envs
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Self-Improving Oversight for AI Customer Support

A three-layer nested RL system that automatically optimizes customer support agents. Layer 0 defines what "good" looks like, Layer 1 finds the best agent instructions, and Layer 2 stress-tests agents against 100 diverse simulated customers.

## Architecture

![Architecture](assets/architecture.png)

### Layer 0 — Reward Architect

Defines the reward function for a given domain. Scores conversations on four axes:

| Signal | Reward |
|---|---|
| Intent accuracy | +50 / -50 |
| Efficiency (fewer turns) | +20 → -5/turn |
| Injection resistance | +40 / -100 |
| API correctness | +20 / -30 |

Domain-pluggable: swap "banking" for "telecom" and a new RL environment is created automatically.

### Layer 1 — Prompt Optimizer (GRPO + Unsloth + TRL)

Trains a prompt-generator model (Qwen 2.5 3B, 4-bit LoRA) to produce optimal system prompts for the voice agent. Each GRPO iteration:

1. Sample N candidate prompts
2. Run K conversations per candidate in Layer 2
3. Score via `reward_fn`
4. Reinforce the best-performing prompts

### Layer 2 — Multi-Agent Arena (OpenEnv 0.2.1)

Simulates multi-turn customer support conversations:

- **Customer** — Llama 3.1 8B with hidden intent + persona (100 personas: 3 intents, 5 personalities, varying social engineering)
- **Voice Agent** — Uses the system prompt from Layer 1
- **API** — Verify, lookup, block actions

Episodes end when the agent classifies intent (success/fail), max turns hit, or the agent leaks unauthorized info.

## Quick Start

```bash
# Install
pip install -e ".[train]"

# Train (requires GPU)
python -m layer1.train --config config.yaml

# Evaluate a prompt
python -m layer1.train --mode eval --prompt "You are a helpful bank agent."

# Run A/B test
python -m scripts.ab_test
```

A Colab notebook is also available at [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

## Stack

| Component | Tool |
|---|---|
| Prompt generator | `unsloth/Qwen2.5-3B-Instruct` (LoRA 4-bit) |
| RL algorithm | TRL `GRPOTrainer` (GRPO) |
| Customer simulator | `meta-llama/Llama-3.1-8B-Instruct` |
| Voice agent | `meta-llama/Llama-3.1-8B-Instruct` |
| Environment | OpenEnv 0.2.1 |
| Tracking | Supabase (conversations, checkpoints, rewards) |
| UI | Gradio |

## Project Structure

```
layer0/          Reward function (domain-pluggable)
layer1/          GRPO training loop + Supabase upload
layer2/          Conversation environment + customer simulator
personas/        100 generated customer personas
scripts/         A/B test utility
notebooks/       Colab training notebook
app.py           Gradio dashboard
config.yaml      All parameters (single source of truth)
```
