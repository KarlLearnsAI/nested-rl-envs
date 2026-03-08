#!/bin/bash
# ============================================================
# Training startup script for HF Spaces / Colab / Northflank
#
# Usage:
#   ./train.sh                          # full run from config.yaml
#   ./train.sh --steps 5 --episodes 3  # quick smoke test
#   HF_TOKEN=hf_xxx ./train.sh         # with inline token
# ============================================================

set -e

echo "============================================================"
echo "  Nested RL Envs — GRPO Training"
echo "  Team: Ludes Magnus"
echo "============================================================"

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set."
    echo "Set it via: export HF_TOKEN=hf_xxx"
    exit 1
fi

# Install training dependencies if not already installed
if ! python -c "import unsloth" 2>/dev/null; then
    echo "Installing training dependencies..."
    pip install -q -e ".[train]"
fi

# Run training
echo "Starting GRPO training..."
python -m layer1.train "$@"
