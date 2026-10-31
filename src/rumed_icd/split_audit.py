"""Privacy-limited record overlap audit; fingerprints do not identify patients or sites."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path

from rumed_icd.data import SPLITS, Record, load_split


def normalize_text(text: str) -> str:
    """Use NFKC, casefold and collapsed whitespace; keep punctuation and word order."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _groups(records: Sequence[Record], key: Callable[[Record], str]) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return groups


def audit_splits(splits: Mapping[str, Sequence[Record]]) -> dict:
    """Count repeated IDs/text and conflicting labels without exporting IDs or complaint text.

    A conflicting key has more than one code across its records. Exact text means the
    loader's stripped text. Normalized text is only a lexical diagnostic, not semantic or
    patient linkage. Shared keys count distinct strings; record counts include all matches.
    """
    keys = {
        "idx": lambda r: r.idx,
        "exact_text": lambda r: r.text,
        "normalized_text": lambda r: normalize_text(r.text),
    }
    indexed = {
        name: {kind: _groups(records, key) for kind, key in keys.items()}
        for name, records in splits.items()
    }
    within, pairs = {}, {}
    for name, records in splits.items():
        within[name] = {"n": len(records), "labels": len({r.code for r in records})}
        for kind, groups in indexed[name].items():
            duplicates = [group for group in groups.values() if len(group) > 1]
            within[name][kind] = {
                "duplicate_keys": len(duplicates),
                "records_in_duplicate_groups": sum(map(len, duplicates)),
                "excess_records": sum(len(group) - 1 for group in duplicates),
                "conflicting_label_keys": sum(
                    len({r.code for r in group}) > 1 for group in duplicates
                ),
            }
    for left, right in combinations(splits, 2):
        pair = pairs[f"{left}:{right}"] = {}
        for kind in keys:
            a, b = indexed[left][kind], indexed[right][kind]
            shared = a.keys() & b.keys()
            pair[kind] = {
                "shared_keys": len(shared),
                "left_records": sum(len(a[key]) for key in shared),
                "right_records": sum(len(b[key]) for key in shared),
                "conflicting_label_keys": sum(
                    len({r.code for r in a[key] + b[key]}) > 1 for key in shared
                ),
            }
    train_labels = {r.code for r in splits.get("train", [])}
    coverage = {
        name: {
            "unseen_in_train": sorted({r.code for r in records} - train_labels),
            "unseen_records": sum(r.code not in train_labels for r in records),
        }
        for name, records in splits.items()
        if name != "train"
    }
    return {
        "within": within,
        "pairs": pairs,
        "label_coverage": coverage,
        "limits": "Record/lexical overlap only. No patient, site or clinical independence proof.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {name: load_split(name, args.raw_dir) for name in SPLITS}
    result = audit_splits(splits)
    result["input_sha256"] = {
        f"{name}_v1.jsonl": hashlib.sha256(
            (args.raw_dir / f"{name}_v1.jsonl").read_bytes()
        ).hexdigest()
        for name in SPLITS
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
