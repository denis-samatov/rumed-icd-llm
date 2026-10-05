# rumed-icd-llm

Prompting vs RAG for ICD-10 coding of Russian clinical complaints, with LoRA fine-tuning and vLLM serving planned. Implemented methods are evaluated on the same public test split. Local numbers in this README come from `results/`; the paper baseline is an external reference.

**Status:** stages A and B are done. Method 0 reproduces the paper baseline, and methods 1–2 have been run on the DeepSeek API. Method 3 (LoRA) and serving are next.

## Task and data

[RuMedTop3](https://github.com/sb-ai-lab/MedBench) (RuMedBench, [arXiv:2201.06499](https://arxiv.org/abs/2201.06499)) asks the model to predict the ICD-10 code from free-text patient complaints. Metrics are Hit@1 and Hit@3.

| Item | Source | License |
|---|---|---|
| Benchmark code and splits | sb-ai-lab/MedBench | Apache-2.0 |
| Underlying records | RuMedPrime, [Zenodo 5765873](https://zenodo.org/records/5765873) | CC BY 3.0 |

The data is not stored in this repository. `scripts/download_data.sh` downloads into a staging directory, verifies the committed `data/raw/SHA256SUMS`, and then installs the files. It refuses changed upstream data without replacing the checksum manifest. Loading a split also verifies its checksum.

## Methods

| # | Method | Status |
|---|---|---|
| 0 | TF-IDF (word + char n-grams) + logistic regression | implemented. It is a harness sanity check against the paper's feature-based baseline (test Hit@1 49.76 / Hit@3 72.75) |
| 1 | Zero-shot and few-shot prompting (15 fixed random training cases) | implemented on the DeepSeek API (`deepseek-flash`, temperature 0, thinking disabled); planned on the same open-weight base model as method 3 |
| 2 | RAG: the 15 most similar training cases in the prompt (TF-IDF char n-gram cosine, retrieval over train only) | implemented, as for method 1 |
| 3 | LoRA / QLoRA SFT (PEFT, TRL) | planned |
| 4 | vLLM serving, fp16 vs AWQ: p50/p95 latency, tokens/s, cost per 1k requests | planned |

Every result reports a 95% bootstrap confidence interval and the rate of invalid ICD-10 codes.

## Run

```bash
scripts/download_data.sh
uv run pytest
uv run python -m rumed_icd.evaluate --method tfidf --split dev
# LLM methods need DEEPSEEK_API_KEY in the environment or in .env (gitignored)
uv run python -m rumed_icd.evaluate --method rag --split dev
```

LLM responses are cached in `results/cache/`, keyed by example and prompt hash. Completed responses are written as they arrive, including when another request fails. A fully cached run requires no API key. A lost response before it is cached can still require a new request; concurrent evaluator processes sharing one cache are unsupported.

Use `--output-dir /tmp/rumed-rerun` to preserve the committed metric files during verification. `--limit` must be positive and writes a separate smoke result; it is not a full-split score.

## Results

Data files are pinned by `data/raw/SHA256SUMS`. Splits: train 4,690, dev 848, test 822; 105 ICD-10 codes appear in train.

![Hit@1 and Hit@3 on the RuMedTop3 test split](docs/results_test.png)

Test split, n = 822. Dev results are in `results/*_dev.json` and show the same ranking. To regenerate the chart, run `uv run --with matplotlib python scripts/plot_results.py`.

| Method | Hit@1 (95% CI) | Hit@3 (95% CI) | Top-1 outside label set | API cost, full test run |
|---|---|---|---|---|
| Paper, feature-based | 49.76 | 72.75 | — | — |
| 0 · TF-IDF + LR | 49.03 (45.62–52.43) | 72.63 (69.46–75.67) | 0% | — |
| 1 · zero-shot, `deepseek-flash` | 33.21 (30.05–36.37) | 50.12 (46.96–53.53) | 4.0% | ≤ $0.07 |
| 1 · few-shot, 15 fixed cases | 33.33 (30.29–36.62) | 52.55 (49.27–55.84) | 3.0% | ≤ $0.08 |
| 2 · RAG, 15 nearest training cases | 47.81 (44.40–51.22) | 72.26 (69.34–75.30) | 1.1% | ≤ $0.34 |

Costs are upper bounds at the recorded peak-hour prices, computed from the token usage the API reported. They estimate one full evaluation, including reused cached responses, rather than the amount billed for a replay. New runs report `new_requests` and `reused_responses` separately. These API costs do not measure vLLM serving costs.

**What this shows so far**

- **Method 0 is a harness sanity check.** Its score is close to the paper's feature-based baseline. Its hyperparameters (C = 10, word 1–2-grams, char 2–5-grams) were fixed before the first run and were not tuned on test. Agreement of aggregate scores does not establish exact reproduction of the paper's implementation.
- **A general-purpose API LLM without dataset examples is about 16 points below a linear classifier on Hit@1.** The labels follow the dataset's own coding practice: a few codes such as M54, I11 and G54 dominate. That practice is not recoverable from general ICD-10 knowledge.
- **Fixed few-shot examples barely help in the recorded run.** Retrieved nearest cases (RAG) close the gap to TF-IDF; the observed RAG score is slightly lower. Overlapping individual confidence intervals do not establish equivalence or test the paired difference. Per-example predictions and a paired comparison are needed for that conclusion.
- **This sets the bar for LoRA (method 3).** Fine-tuning has to beat about 49 / 73 to justify itself over a classifier that trains in seconds.

## Evidence limits

This benchmark is research software for dataset-specific coding, not a diagnostic service. Its record split does not establish patient-disjoint validation, performance at another institution, or clinical utility. No private institutional data is used here.

The API runs use a different model from the planned open-weight LoRA experiment. A controlled fine-tuning comparison must rerun prompting and RAG with the same base model, select hyperparameters on dev, and evaluate test only after that choice is frozen. Training, GPU latency, throughput, quantization and serving cost are not yet demonstrated.

Committed LLM metric summaries do not contain the response cache. Anyone can inspect them, but independent score reconstruction requires the corresponding per-example responses or a new API run. Offline tests mock all paid calls.
