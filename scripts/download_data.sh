#!/usr/bin/env bash
# Download RuMedTop3 (github.com/sb-ai-lab/MedBench, Apache-2.0; data: RuMedPrime, CC BY 3.0)
# into data/raw/ and verify the committed SHA-256 checksums before replacing files.
set -euo pipefail

BASE="https://raw.githubusercontent.com/sb-ai-lab/MedBench/main/data/RuMedTop3"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"
STAGE="$(mktemp -d "$DEST/.download.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
cp "$DEST/SHA256SUMS" "$STAGE/SHA256SUMS"
for split in train dev test; do
  curl -fsSL "$BASE/${split}_v1.jsonl" -o "$STAGE/${split}_v1.jsonl"
done
(cd "$STAGE" && shasum -a 256 -c SHA256SUMS)
for split in train dev test; do
  mv "$STAGE/${split}_v1.jsonl" "$DEST/${split}_v1.jsonl"
done
cat "$DEST/SHA256SUMS"
