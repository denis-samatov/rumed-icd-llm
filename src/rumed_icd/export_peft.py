"""Export an MLX LoRA checkpoint as PEFT safetensors for vLLM-Metal."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def convert_weights(weights: Mapping[str, np.ndarray], rank: int) -> dict[str, np.ndarray]:
    if not weights or rank < 1:
        raise ValueError("weights must be nonempty and rank positive")
    converted = {}
    for name, value in weights.items():
        module, leaf = name.rsplit(".", 1)
        if leaf not in ("lora_a", "lora_b") or value.ndim != 2:
            raise ValueError(f"unexpected adapter tensor: {name}")
        rank_axis = 1 if leaf == "lora_a" else 0
        if value.shape[rank_axis] != rank:
            raise ValueError(f"rank mismatch: {name}")
        partner = module + (".lora_b" if leaf == "lora_a" else ".lora_a")
        if partner not in weights:
            raise ValueError(f"missing paired tensor: {partner}")
        peft_name = f"base_model.model.{module}.{leaf[:-1]}{leaf[-1].upper()}.weight"
        converted[peft_name] = np.ascontiguousarray(value.T)
    return converted


def export(adapter: Path, output: Path) -> None:
    import mlx.core as mx
    from safetensors.numpy import save_file

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output must be empty; refusing to overwrite an exported adapter")
    log = json.loads(adapter.with_suffix(".json").read_text())
    config = log["config"]
    mx.set_default_device(mx.cpu)
    weights = {name: np.array(value) for name, value in mx.load(str(adapter)).items()}
    converted = convert_weights(weights, config["rank"])
    modules = sorted({name.split(".")[-3] for name in converted})
    layers = sorted({int(name.split(".layers.", 1)[1].split(".", 1)[0]) for name in weights})
    peft_config = {
        "base_model_name_or_path": "Qwen/Qwen3-8B", "peft_type": "LORA",
        "task_type": "CAUSAL_LM", "r": config["rank"],
        # MLX applies scale directly; PEFT uses alpha / rank.
        "lora_alpha": config["scale"] * config["rank"],
        "lora_dropout": config["dropout"], "bias": "none", "inference_mode": True,
        "target_modules": modules, "layers_to_transform": layers,
    }
    output.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output / "adapter_model.safetensors"))
    (output / "adapter_config.json").write_text(json.dumps(peft_config, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "model": log["model"], "model_revision": log["model_revision"],
        "completed_steps": log["completed_steps"], "status": log["status"],
        "conversion": "transpose MLX A/B; PEFT alpha = MLX scale * rank",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.adapter, args.output)


if __name__ == "__main__":
    main()
