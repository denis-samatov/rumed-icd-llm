# Reproduction acceptance boundaries

## Dataset transfer and clinical use

This repository evaluates public benchmark records. It does not establish patient-disjoint,
site-disjoint or prospective clinical performance. Before any real clinical study, obtain
approved records with patient/site identifiers, de-duplicate across groups, freeze grouped
train/dev/test IDs before tuning, record intended use and abstention, and report paired
record and patient-cluster uncertainty. Missing identifiers must remain an explicit limitation.
No real institutional data or clinical qualification is created by the offline test suite.

## API evidence handoff

The published DeepSeek response caches are not available in this clone. A future authorized
handoff should contain only idx, parsed top-three codes, method/model, prompt hash, token usage,
and source-cache SHA-256, plus a checksum manifest and dataset attribution. Do not include
complaint text, prompts, credentials or unreviewed response prose. Verify IDs/order against
the checksum-validated split, independently recompute metrics, and compare paired methods.
A new paid request is a new experiment, not reconstruction of missing historical responses.

## Validation

From a checkout with the maintained CPU execution profile installed (see below):

```bash
python -m pytest -q
python scripts/verify_qwen3_results.py
python scripts/verify_kev_results.py
```

Use `--output-dir /tmp/rumed-rerun` for metrics **and cache** isolation. Separate processes
must use separate output directories. The default `results/` preserves existing cache paths.
CPU prediction replay verifies saved calculations, not model inference or serving throughput.

Package code can be built with `uv build`; dataset/prediction verification still needs the
checkout resources. A wheel import smoke is not a resource-complete benchmark reproduction.

## Separate execution profile (2026-10-29)

`requirements-execution.in` is the maintained input; `requirements-execution.txt` is generated
with `uv pip compile requirements-execution.in --python-version 3.13 --generate-hashes --output-file requirements-execution.txt`.
It is a new Python 3.13 software-validation environment, not the historical scientific environment.
Create a local venv, then install with `python -m pip install --require-hashes -r requirements-execution.txt`.
Do not edit the generated lock by hand. OSV exact-version scanning covers the resolved public
packages; it does not establish exploit reachability or guarantee that undisclosed issues are absent.
Historical locks, predictions and numerical results remain unchanged. A new experimental result
requires its own inputs, source revision, environment and numerical qualification.

For CPU replay use the new profile and `python -m pytest tests -q`,
`PYTHONPATH=src python scripts/verify_qwen3_results.py`, and
`PYTHONPATH=src python scripts/verify_kev_results.py`. The wheel contains package code;
checkout `data/` and `results/` are still needed for independent saved-prediction replay.
`requirements-execution-mlx.in/.txt` separately resolves the optional MLX stack with patched
certificate/file-lock dependencies. Library imports do not qualify model inference, throughput,
adapter training or numerical parity; those require explicit model/data execution.

This profile was resolved on macOS arm64 for Python 3.13. It is not a universal lock for
all operating systems, Python versions or accelerators; regenerate and qualify another target.
