#!/usr/bin/env bash
# Download RuMedTop3 (github.com/sb-ai-lab/MedBench, Apache-2.0; data: RuMedPrime, CC BY 3.0)
# into data/raw/ and record SHA-256 checksums so every run is tied to exact files.
set -euo pipefail

BASE="https://raw.githubusercontent.com/sb-ai-lab/MedBench/main/data/RuMedTop3"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"
for split in train dev test; do
  curl -fsSL "$BASE/${split}_v1.jsonl" -o "$DEST/${split}_v1.jsonl"
done
(cd "$DEST" && shasum -a 256 ./*_v1.jsonl > SHA256SUMS)
cat "$DEST/SHA256SUMS"
