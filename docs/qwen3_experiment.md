# Qwen3-8B: training, evaluation and local serving

[Benchmark overview and shared evidence limits](../README.md). Run commands from the repository root.

## Training on Apple Silicon

Use a native arm64 Python with `uv sync --locked --extra mlx`. The base model is
`mlx-community/Qwen3-8B-4bit`, revision `545dc4251c05440727734bcd94334791f6ab0192`.
Set `RUMED_MODEL_PATH` to that downloaded snapshot for offline operation.

```bash
# A bounded pipeline check, written separately from the full training run.
uv run --locked --extra mlx python -m rumed_icd.train_lora \
  --max-steps 5 --dev-loss-examples 8 --output adapters/pilot/lora_qwen3_8b.safetensors
# One complete epoch over train; dev is used only to measure loss.
uv run --locked --extra mlx python -m rumed_icd.train_lora
# Same model and scoring rule; inspect dev before freezing test evaluation.
uv run --locked --extra mlx python -m rumed_icd.evaluate --method local_zero_shot --split dev
uv run --locked --extra mlx python -m rumed_icd.evaluate --method local_lora --split dev
# After freezing choices, evaluate the held-out test split.
uv run --locked --extra mlx python -m rumed_icd.evaluate --method local_zero_shot --split test
uv run --locked --extra mlx python -m rumed_icd.evaluate --method local_lora --split test
# Also available: local_few_shot and local_rag, retrieval over train only.
```

The adapter trains the last 16 blocks, rank 8, MLX scale 20, learning rate 1e-4,
one epoch, seed 0. Microbatch 1 with accumulation 4 and gradient checkpointing
keeps memory bounded. Loss is computed in float32 on the three observed code tokens
and the end-of-turn token only. No additional gold labels are invented.
Every 10 optimizer steps, adapter weights and a progress JSON are saved atomically.
These are weight checkpoints; they do not restore optimizer state for an exact resume.
Existing adapter files are not overwritten when starting a run.

Local predictions rank all 105 training codes by three-token log-likelihood.
They use the same non-thinking prompt for base and LoRA. This forced-choice rule
differs from the DeepSeek JSON generation experiment, and makes label validity
automatic. Cached results include the prompt, model revision and adapter hash.
BF16 execution can change rankings of close candidates between cached and direct
prefill; numerical checks must accompany any full result. A short dev sample and
loss reduction do not establish improvement on the full held-out benchmark.

## Serve the adapter with vLLM-Metal

The community [vLLM-Metal plugin](https://github.com/vllm-project/vllm-metal)
uses an independent environment because it pins a different MLX build:

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python \
  'https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl' \
  'https://github.com/vllm-project/vllm-metal/releases/download/v0.30.0/vllm_metal-0.30.0-cp312-cp312-macosx_15_0_arm64.whl'
uv run --locked --extra mlx python -m rumed_icd.export_peft \
  --adapter adapters/lora_qwen3_8b.safetensors --output adapters/qwen3-peft
RUMED_VLLM_BIN="$PWD/.venv-vllm/bin/vllm" bash scripts/serve_local.sh "$PWD/adapters/qwen3-peft"
```

The exporter transposes MLX matrices into PEFT orientation and sets alpha to
`scale * rank`. The base quantization is preserved. The API binds to
`http://127.0.0.1:8001/v1` by default (`RUMED_PORT=8002` selects another port);
request `model: "rumed"` to use the adapter and
`model: "qwen3-base"` for the base. Pass `chat_template_kwargs: {"enable_thinking": false}`
in chat requests to match the training prompt. Serving generates text; the local
benchmark's label-set scorer is a separate evaluation path.

`scripts/run_local.sh` runs the full epoch, exports and starts the API in sequence.
It requires `RUMED_VLLM_BIN` to identify the independent serving environment.
Training and serving are sequential to avoid two 8B models sharing 16 GB RAM.

## One epoch: full held-out evaluation

Measured on 2026-10-06 using MLX 0.32.3 / mlx-lm 0.32.0 on an Apple M2 Pro
with 16 GB RAM. Base and LoRA use the same pinned 4-bit model revision,
non-thinking prompt and scoring rule: sum of log-probabilities of three code
tokens, ranking all 105 train labels; the end-of-turn token is not scored.
Weights and hyperparameters were fixed before this evaluation, including test.

![Qwen3 base and one-epoch LoRA test Hit@1 and Hit@3 with 95% confidence intervals; TF-IDF is a previously measured reference](results_qwen3_test.png)

Regenerate this separate Qwen3 chart from the committed metric files with
`uv run --with matplotlib python scripts/plot_qwen3_results.py`.

| Split | Model | Hit@1 (95% CI) | Hit@3 (95% CI) |
|---|---|---|---|
| dev, n=848 | Qwen3 base, zero-shot | 10.38 (8.49–12.62) | 23.11 (20.40–25.94) |
| dev, n=848 | Qwen3 + LoRA | 35.85 (32.78–39.03) | 57.78 (54.60–61.20) |
| test, n=822 | Qwen3 base, zero-shot | 7.66 (5.96–9.49) | 17.27 (14.60–19.71) |
| test, n=822 | Qwen3 + LoRA | **35.64 (32.36–38.93)** | **60.10 (56.69–63.38)** |

Paired LoRA minus base differences on test are **+27.98 percentage points**
for Hit@1 (95% CI +24.33 to +31.75) and **+42.82 points** for Hit@3
(+38.69 to +46.96), using 2,000 bootstrap resamples, seed 0.
LoRA improves its base but does **not** beat the previously reproduced TF-IDF
result of 49.03 / 72.63 on the same test bytes. TF-IDF was not rerun during
this Qwen3 evaluation. Local few-shot and RAG have not been measured.

Training completed 1,173 optimizer steps over 4,690 train records, one epoch,
in 23,576 seconds (6 h 32 min 56 s), with 7.075 GiB peak MLX memory.
Loss on the fixed 32-example dev subset decreased from 3.1931 to 0.5376;
this subset loss is separate from the full-split accuracy above.
The checkpoint contains weights only, without optimizer state.

[Training metadata](../results/qwen3_training.json) and
[evaluation provenance and paired statistics](../results/qwen3_summary.json)
record the model revision, adapter SHA-256, dataset and prediction hashes.
The model weights are not committed. The prediction files contain benchmark
record IDs, gold codes and base/LoRA top-three codes, without complaint text:
[dev](../results/qwen3_predictions_dev.jsonl), [test](../results/qwen3_predictions_test.jsonl).
These dataset-derived labels retain the RuMedPrime CC BY 3.0 attribution
linked in the [benchmark overview](../README.md#task-and-data); repository code uses its own license.

Independently reconstruct the published metrics and paired confidence
intervals on CPU, without model weights, an API key or a model download:

```bash
uv sync --locked
uv run python scripts/verify_qwen3_results.py
```

This verifies the saved predictions and aggregate calculations; rerunning
model inference requires the optional MLX environment and a trained adapter.
BF16 differences between cached and direct scoring may reorder close
candidates. Direct evaluation of all 105 codes agreed on top-three rankings
for two checked dev cases; this does not establish agreement on all records.
The serving API was also checked with the full adapter, but that smoke is
separate from classification accuracy and is not a latency benchmark.
