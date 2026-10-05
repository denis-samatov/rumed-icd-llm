#!/usr/bin/env bash
# Method 3 and the same-model baselines on Apple Silicon (MLX). Long-running: hours.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
uv run --extra mlx python -m rumed_icd.train_lora
for m in local_lora local_zero_shot local_few_shot local_rag; do
  uv run --extra mlx python -m rumed_icd.evaluate --method "$m" --split test
done
