"""RuMedTop3 loading. Source: github.com/sb-ai-lab/MedBench (Apache-2.0), data from RuMedPrime
(Zenodo 5765873, CC BY 3.0). Each JSONL line: {"idx": str, "symptoms": str, "code": str}."""

from __future__ import annotations

import hashlib
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
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            missing = {"idx", "symptoms", "code"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
            for name in ("idx", "symptoms", "code"):
                if not isinstance(row[name], str) or not row[name].strip():
                    raise ValueError(f"{path}:{line_no}: {name} must be a nonempty string")
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
    manifest = raw_dir / "SHA256SUMS"
    expected = None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if name.lstrip("*").removeprefix("./") == path.name:
            expected = digest
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{path}: checksum differs from {manifest}; refusing dataset drift")
    return load_jsonl(path)
