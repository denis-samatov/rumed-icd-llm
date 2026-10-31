# Public-record CPU qualification (2026-10-31)

This qualification checks split overlap, a fixed baseline and metric implementations.
It is not a new independent clinical evaluation. The public dev/test records were
already used in the recorded experiments and the earlier software reproduction.
The pre-evaluation protocol SHA-256 is
`50d7cc57aaca75a83d694f15d013ebd998533f709753070dfec87cae499cf4a8`;
the evaluated base was `5eb70676ea37eb1283c998244bfd2eba87da1d30`.
No holdout-driven hyperparameter adjustment, record deletion or label correction was made.

## Input identity and split audit

All three JSONL files passed the committed SHA256SUMS check. The existing source
metadata identifies MedBench/RuMedTop3, derived from RuMedPrime; use the repository's
[reproduction guide](reproduction_acceptance.md) for acquisition and attribution.
This run inspected only that byte-identified public release, not clinical source records.

| Split | Records | Distinct labels | SHA-256 |
| --- | ---: | ---: | --- |
| train | 4690 | 105 | `b185fe85ad4b4346be3180997fa77816b6e4166567560f2ce948428c51eb6b85` |
| dev | 848 | 104 | `6b56b2d916a9da92064c749092d2fc4a1677e17b9bb03978f41b213e99c6d286` |
| test | 822 | 104 | `2be46c07dabe26976b6393f93e6333818665530953e7c22007c6080de3dca507` |

The new `rumed_icd.split_audit` counts distinct shared keys, records involved and
label conflicts for IDs, exact loader text and normalized text. Normalization is
NFKC → casefold → whitespace collapse; punctuation and word order remain intact.
It exports counts/labels, without complaint text or individual fingerprints.

- All three split pairs had **zero** shared IDs, exact texts and normalized texts.
- Train contained **one** duplicate-text group with **two** records and **two
  different labels**. The same group was detected after normalization. Official
  labels/records were retained; this run does not determine which label is correct.
- Dev/test had no within-split duplicate texts under these rules.
- Every dev/test label occurred in train. All 848/822 evaluation records therefore
  fell in the predeclared “no normalized train-text match” descriptive subsets.

The actual raw schema contains only `idx`, `symptoms`, `code`, all strings. No
patient/site identifiers or patient-linkage evidence were available. Zero lexical
matches does not exclude paraphrases, repeated patients or shared institutional
sources. No patient-disjoint, source-disjoint or temporal holdout is established.

## Fixed baseline and independent statistics

The unchanged TF-IDF word/character features and logistic regression used C=10,
seed=0 and train-only fitting. No neural model, API or GPU was used. The runtime
was Python 3.13.13, NumPy 2.5.4, scikit-learn 1.9.1 on macOS arm64, CPU threads=1.

| Split | Top-1 successes | Hit@1 | Record bootstrap 95% CI | Top-3 successes | Hit@3 | Record bootstrap 95% CI |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| dev | 412/848 | 48.58% | 45.17–52.00% | 613/848 | 72.29% | 69.10–75.24% |
| test | 403/822 | 49.03% | 45.62–52.43% | 597/822 | 72.63% | 69.46–75.67% |

An independent scalar membership counter and row-wise bootstrap (2000 resamples,
seed=0) matched the repository's vectorized rates and rounded percentile intervals.
Wilson intervals were also calculated as a separately specified binomial check:
test Top-1 45.62–52.44%, Top-3 69.48–75.57%. Wilson and bootstrap intervals are
not expected to be identical. Both assume independent records; neither accounts
for unavailable patient/site clustering or uncertainty from model selection.

The ID/code-only prediction JSONL SHA-256 was
`6a29d3dcabd976f11b5de7203f44675e2a354b22b8058ae7514a6833b1f30f15`.
Matching scores verify implementation/statistics on this exposed release. They do
not independently confirm clinical utility, neural-model quality, transfer to a
new institution, or generalization to an unseen benchmark.

## Validation and reproduction

Use the maintained core execution environment, then run from the source checkout:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m rumed_icd.split_audit \
  --raw-dir /path/to/checksum-verified/public-data --output /tmp/rumed-split-audit.json
PYTHONPATH=src python scripts/qualify_baseline.py \
  --raw-dir /path/to/checksum-verified/public-data --output-dir /tmp/rumed-cpu-qualification
```

The baseline command requires a fresh output directory and retains input checksums,
runtime, overlap counts, independent statistics and public benchmark IDs/top-3 codes.
It never writes complaint text. Synthetic tests cover normalization, conflicting
labels, duplicated keys, unseen labels, known success counts and invalid alignment.
Keep this diagnostic separate from the existing Qwen/Kev saved-prediction replays;
it does not rerun their training or inference.
