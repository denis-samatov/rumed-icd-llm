"""RuMedTop3 loading. Source: github.com/sb-ai-lab/MedBench (Apache-2.0), data from RuMedPrime
(Zenodo 5765873, CC BY 3.0). Each JSONL line: {"idx": str, "symptoms": str, "code": str}."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "dev", "test")
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass(frozen=True)
class Record:
    idx: str
    text: str
    code: str


def load_jsonl(path: Path) -> list[Record]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"idx", "symptoms", "code"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
            records.append(Record(row["idx"], row["symptoms"].strip(), row["code"].strip()))
    ids = [r.idx for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate idx values")
    return records


def load_split(split: str, raw_dir: Path = RAW_DIR) -> list[Record]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = raw_dir / f"{split}_v1.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/download_data.sh first")
    return load_jsonl(path)
