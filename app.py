"""
HF Spaces Gradio App — Training results & architecture overview
for the Nested RL Environments system.
"""
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
# ── Training episode data (from Supabase export) ──
# Episodes 0–75, sorted by step → episode → created_at
EPISODE_REWARDS = [
    39.7, -20.1, -14.2, 96.6, -60.3, -60.1, 35.8, 96.6, -60.3, -5.1,
    35.8, 41.6, 39.7, 19.9, 90.8, 41.6, 94.7, -60.1, -124.2, 41.6,
    46.7, -56.3, 50.6, 48.6, 101.7, -56.3, 50.6, 48.6, 46.7, -1.3,
    50.6, -11.4, 46.7, -56.3, 50.6, 103.6, 46.7, -56.3, 5.6, -121.4,
    -13.9, 41.3, 100.3, -52.3, 46.1, 96.3, 45.3, -142.3, 91.1, 96.3,
    100.3, -52.3, 46.1, 96.3, 0.3, -52.3, -13.9, 41.3, 45.3, -12.3,
    33.8, 54.8, 91.0, -64.3, 33.8, 54.8, 36.0, 90.7, 33.8, 109.8,
    36.0, -64.3, 33.8, 54.8, 91.0, -19.3,
]
def compute_rolling_avg(rewards, window=35):
    """Compute rolling average with given window size."""
    avgs = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        w = rewards[start : i + 1]
        avgs.append(sum(w) / len(w))
    return avgs
def create_reward_chart():
    """Generate the reward trend chart as a matplotlib figure."""
    episodes = list(range(len(EPISODE_REWARDS)))
    rolling = compute_rolling_avg(EPISODE_REWARDS, window=35)
    # ── Dark theme ──
    bg = "#0a0e17"
    card = "#111827"
    border = "#1e293b"
    cyan = "#22d3ee"
    cyan_dim = (34 / 255, 211 / 255, 238 / 255, 0.15)
    text = "#e2e8f0"
    muted = "#475569"
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=bg)
    ax.set_facecolor(card)
    # Subtle border
    for spine in ax.spines.values():
        spine.set_color(border)
        spine.set_linewidth(0.8)
    # Rolling average line + fill
    ax.plot(episodes, rolling, color=cyan, linewidth=2.8, zorder=3)
    ax.fill_between(
        episodes, rolling, alpha=0.15, color=cyan, zorder=2,
    )
    # Zero reference line
    ax.axhline(y=0, color=border, linewidth=0.8, linestyle="--", zorder=1)
    # Axis styling
    ax.set_xlabel("Episode", color=muted, fontsize=11, fontfamily="monospace")
    ax.set_ylabel("Reward", color=muted, fontsize=11, fontfamily="monospace")
    ax.tick_params(colors=muted, labelsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(False)
    # Title
    ax.set_title(
        "Reward Trend  ·  Episodes 0–75  ·  35-ep Rolling Avg",
        color=text,
        fontsize=14,
        fontfamily="monospace",
        fontweight="bold",
        pad=16,
    )
    fig.tight_layout(pad=1.5)
    return fig
# ── Build the Gradio app ──
with gr.Blocks(
    title="Nested RL Environments — AI Oversight",
    theme=gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
    ),
    css="""
        .gradio-container { background: #0a0e17 !important; }
        .main-header { text-align: center; margin-bottom: 8px; }
        .main-header h1 { color: #e2e8f0; font-family: monospace; }
        .main-header p { color: #64748b; font-family: monospace; font-size: 14px; }
        .section-label { color: #94a3b8 !important; font-family: monospace !important; }
    """,
) as demo:
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>Nested RL Environments</h1>
            <p>Self-Improving Oversight for AI Customer Support · Team Ludes Magnus</p>
        </div>
    """)
    # ── Tab layout ──
    with gr.Tabs():
        # Tab 1: Training Results
        with gr.Tab("Training Results"):
            gr.Markdown(
                "### Reward Trend — GRPO Prompt Optimization",
                elem_classes=["section-label"],
            )
            reward_plot = gr.Plot(value=create_reward_chart(), label="Reward Trend")
            gr.Markdown(
                """
                <div style="color: #64748b; font-family: monospace; font-size: 12px; text-align: right; margin-top: -8px;">
                    Run 20260308_135709 · 300 total episodes · All data authentic
                </div>
                """,
            )
        # Tab 2: Architecture (placeholder for future .png)
        with gr.Tab("Architecture"):
            gr.Markdown("""
# The 3-Layer Architecture
```
┌─────────────────────────────────────────────────────────┐
│  LAYER 0 — Reward Function                              │
│                                                         │
│  Defines what "good" looks like for a conversation:     │
│  • +50  Correct intent classification                   │
│  • +20  Resolved in ≤3 turns (efficiency)               │
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
## Prize Targets
- **Main Track — Statement 4:** Layer 0 generates reward functions → new domain = new RL environment automatically
- **Fleet AI $10k:** Layer 1 provides scalable oversight — add intents, retrain
- **Halluminate $10k:** Layer 2 is a multi-actor environment with 100 diverse adversarial customers
""")
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
