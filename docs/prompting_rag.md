# Prompting and RAG: DeepSeek API experiment

[Benchmark overview and shared evidence limits](../README.md). Run commands from the repository root.

The API uses `deepseek-flash`, temperature 0, with thinking disabled. Few-shot supplies 15 fixed random train cases; RAG retrieves 15 nearest train cases using TF-IDF character n-gram cosine similarity.

## Results and API costs

Data files are pinned by `data/raw/SHA256SUMS`. Splits: train 4,690, dev 848, test 822; 105 ICD-10 codes appear in train.

![Hit@1 and Hit@3 on the RuMedTop3 test split](results_test.png)

Test split, n = 822. Dev results are in `results/*_dev.json` and show the same ranking. To regenerate the chart, run `uv run --with matplotlib python scripts/plot_results.py`.

| Method | Hit@1 (95% CI) | Hit@3 (95% CI) | Top-1 outside label set | API cost, full test run |
|---|---|---|---|---|
| Paper, feature-based | 49.76 | 72.75 | — | — |
| 0 · TF-IDF + LR | 49.03 (45.62–52.43) | 72.63 (69.46–75.67) | 0% | — |
| 1 · zero-shot, `deepseek-flash` | 33.21 (30.05–36.37) | 50.12 (46.96–53.53) | 4.0% | ≤ $0.07 |
| 1 · few-shot, 15 fixed cases | 33.33 (30.29–36.62) | 52.55 (49.27–55.84) | 3.0% | ≤ $0.08 |
| 2 · RAG, 15 nearest training cases | 47.81 (44.40–51.22) | 72.26 (69.34–75.30) | 1.1% | ≤ $0.34 |

The invalid top-1 code rate is 0% for each measured method in this table. A valid
ICD-10 code can still fall outside the dataset's 105-code train label set, as shown
in the separate column above.

Costs are upper bounds at the recorded peak-hour prices, computed from the token usage the API reported. They estimate one full evaluation, including reused cached responses, rather than the amount billed for a replay. New runs report `new_requests` and `reused_responses` separately. These API costs do not measure vLLM serving costs.

## Findings

- **Method 0 is a harness sanity check.** Its score is close to the paper's feature-based baseline. Its hyperparameters (C = 10, word 1–2-grams, char 2–5-grams) were fixed before the first run and were not tuned on test. Agreement of aggregate scores does not establish exact reproduction of the paper's implementation.
- **A general-purpose API LLM without dataset examples is about 16 points below a linear classifier on Hit@1.** The labels follow the dataset's own coding practice: a few codes such as M54, I11 and G54 dominate. That practice is not recoverable from general ICD-10 knowledge.
- **Fixed few-shot examples barely help in the recorded run.** Retrieved nearest cases (RAG) close the gap to TF-IDF; the observed RAG score is slightly lower. Overlapping individual confidence intervals do not establish equivalence or test the paired difference. Per-example predictions and a paired comparison are needed for that conclusion.
- The separate [Qwen3 experiment](qwen3_experiment.md) measures base versus LoRA using the same local scoring rule.
