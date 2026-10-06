"""Train the LoRA adapter for method 3. Usage: uv run --extra mlx python -m rumed_icd.train_lora"""

from __future__ import annotations

import argparse
from pathlib import Path

from rumed_icd.data import load_split
from rumed_icd.local_llm import ROOT, train_lora


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "adapters" / "lora_qwen3_8b.safetensors")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dev-loss-examples", type=int, default=32)
    args = parser.parse_args()
    log = train_lora(load_split("train"), load_split("dev"),
                     args.output, n_dev_loss=args.dev_loss_examples, max_steps=args.max_steps)
    print({k: v for k, v in log.items() if k != "steps"})


if __name__ == "__main__":
    main()
