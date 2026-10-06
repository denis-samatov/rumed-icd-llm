#!/usr/bin/env bash
# An isolated Python 3.12 vLLM-Metal environment is required; see README.
set -euo pipefail
cd "$(dirname "$0")/.."
adapter=${1:?Usage: serve_local.sh PATH_TO_PEFT_ADAPTER}
test -f "$adapter/adapter_config.json"
test -f "$adapter/adapter_model.safetensors"
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
exec "${RUMED_VLLM_BIN:-vllm}" serve "${RUMED_MODEL_PATH:-mlx-community/Qwen3-8B-4bit}" \
  --revision 545dc4251c05440727734bcd94334791f6ab0192 \
  --served-model-name qwen3-base --host 127.0.0.1 --port "${RUMED_PORT:-8001}" \
  --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.55 --kv-cache-memory-bytes 536870912 \
  --enable-lora --max-loras 1 --max-lora-rank 16 --lora-modules "rumed=$adapter"
