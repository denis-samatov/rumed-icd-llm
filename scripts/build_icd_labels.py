"""Generate Russian train-label descriptions from a pinned public ICD-10 directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

from rumed_icd.data import load_split

REVISION = "519b65e608d7f15beb947cb0d7bc14071018a710"
URL = (f"https://raw.githubusercontent.com/ak4nv/mkb10/{REVISION}/"
       "resources/1.2.643.5.1.13.13.11.1005_2.27.csv")
ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "3b0a2ff314b3a1e1489338ae9e83c15fbdf4f98250f7b886c27edb60ef507509"


def build(raw: bytes, codes: list[str]) -> dict:
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("pinned ICD directory bytes changed")
    directory = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";"):
        code, title = row["MKB_CODE"].strip(), row["MKB_NAME"].strip()
        if code in directory and directory[code] != title:
            raise ValueError(f"conflicting titles for {code}")
        directory[code] = title
    missing = set(codes) - directory.keys()
    if missing or any(not directory[c] for c in codes):
        raise ValueError(f"missing ICD titles: {sorted(missing)}")
    return {"source_url": URL, "source_revision": REVISION,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_repository_license": "MIT (see docs/ICD_LABELS_LICENSE.txt)",
            "directory": "Russian ICD-10 directory 1.2.643.5.1.13.13.11.1005, v2.27",
            "labels": [{"code": c, "description": directory[c]} for c in codes]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, help="use a downloaded source instead of network")
    ap.add_argument("--output", type=Path, default=ROOT / "data/icd_labels_ru.json")
    args = ap.parse_args()
    if args.csv:
        raw = args.csv.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            raw = response.read()
    codes = sorted({r.code for r in load_split("train")})
    result = build(raw, codes)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")
    print(f"{len(codes)} descriptions: {args.output}")


if __name__ == "__main__":
    main()
