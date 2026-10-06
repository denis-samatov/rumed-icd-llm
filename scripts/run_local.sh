#!/usr/bin/env bash
# One train epoch, export the trained adapter, then serve locally.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
python=${RUMED_PYTHON:-.venv/bin/python}
"$python" -u -m rumed_icd.train_lora
"$python" -m rumed_icd.export_peft \
  --adapter adapters/lora_qwen3_8b.safetensors --output adapters/qwen3-peft
exec bash scripts/serve_local.sh "$(pwd)/adapters/qwen3-peft"
