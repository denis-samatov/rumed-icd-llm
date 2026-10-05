"""Train the LoRA adapter for method 3. Usage: uv run --extra mlx python -m rumed_icd.train_lora"""

from __future__ import annotations

from rumed_icd.data import load_split
from rumed_icd.local_llm import ROOT, train_lora


def main() -> None:
    log = train_lora(load_split("train"), load_split("dev"),
                     ROOT / "adapters" / "lora_qwen3_8b.safetensors")
    print({k: v for k, v in log.items() if k != "steps"})


if __name__ == "__main__":
    main()
