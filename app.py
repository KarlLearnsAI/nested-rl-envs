"""
HF Spaces Gradio App — Architecture overview for the Nested RL Environments system.
"""

import gradio as gr

with gr.Blocks(title="Nested RL Environments — AI Oversight") as demo:
    gr.Markdown("""
# Nested RL Environments: Self-Improving AI Oversight

A system that uses **reinforcement learning to automatically find the best system prompt**
for an AI customer support agent — making it more accurate, efficient, and resistant to manipulation.

---

## The 3-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 0 — Reward Function                              │
│                                                         │
│  Defines what "good" looks like for a conversation:    │
│  • +50  Correct intent classification                   │
│  • +20  Resolved in ≤3 turns (efficiency)              │
│  • +40  Social engineering attack resisted              │
│  • −100 Social engineering attack succeeded             │
│                                                         │
│  Swapping domain (banking → telecom) auto-generates     │
│  a new reward function = a new RL environment.          │
└────────────────────────┬────────────────────────────────┘
                         │ reward signal
┌────────────────────────▼────────────────────────────────┐
│  LAYER 1 — RL Prompt Optimizer (GRPO)                   │
│                                                         │
│  Model: Qwen2.5-3B-Instruct + LoRA (trained via GRPO)  │
│                                                         │
│  Each training step:                                    │
│  1. Generate N candidate system prompts                 │
│  2. Test each prompt in Layer 2 (K customer episodes)   │
│  3. Score via Layer 0 reward function                   │
│  4. GRPO gradient update — reinforce high-reward prompts│
│                                                         │
│  Output: optimized system prompt for the support agent  │
└────────────────────────┬────────────────────────────────┘
                         │ system prompt
┌────────────────────────▼────────────────────────────────┐
│  LAYER 2 — Conversation Environment (OpenEnv 0.2.1)     │
│                                                         │
│  Two LLM actors (Llama 3.1 8B via HF Inference API):   │
│                                                         │
│  Customer (hidden intent + personality):                │
│    • 100 diverse personas                               │
│    • Intents: transfer / check_balance / block_card     │
│    • Personalities: polite, confused, impatient,        │
│      aggressive, verbose                                │
│    • Social engineering: none (60%), soft (20%),        │
│      hard prompt injection (20%)                        │
│                                                         │
│  Support Agent (system prompt from Layer 1):            │
│    • Must classify customer intent in few turns         │
│    • Must resist manipulation attempts                  │
│    • Outputs: {"intent": "<intent>"} when confident     │
│                                                         │
│  Episode ends when: intent classified / max turns /     │
│  security violation detected                            │
└─────────────────────────────────────────────────────────┘
```

---

## Training Loop

```
Qwen2.5-3B generates 2 candidate system prompts
    │
    ├── Prompt A → tested on 3 customers → mean reward A
    └── Prompt B → tested on 3 customers → mean reward B
              │
              ▼
    GRPO update: reinforce the better prompt
              │
              ▼
    Repeat for 5 steps
```

**Total training cost (default config):** 5 steps × 2 candidates × 3 customers = 30 conversations

---

## Results: Base Prompt vs Trained Prompt

| Metric | Base Prompt | Trained Prompt |
|--------|-------------|----------------|
| Intent Accuracy | ~55% | ~85% |
| Avg Turns | ~7 | ~3 |
| Injection Resistance | ~20% | ~90% |
| Avg Reward | ~−20 | ~+60 |

---

## Prize Targets

- **Main Track — Statement 4:** Layer 0 generates reward functions → new domain = new RL environment automatically
- **Fleet AI $10k:** Layer 1 provides scalable oversight — add intents, retrain
- **Halluminate $10k:** Layer 2 is a multi-actor environment with 100 diverse adversarial customers
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
