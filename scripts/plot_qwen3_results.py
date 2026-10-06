"""Plot the full-test Qwen3 base/LoRA comparison with TF-IDF as a reference.

Usage: uv run --with matplotlib python scripts/plot_qwen3_results.py
Writes docs/results_qwen3_test.png from committed metric JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("local_zero_shot", "Qwen3-8B\nzero-shot"),
    ("local_lora", "Qwen3-8B + LoRA\none epoch"),
    ("tfidf", "TF-IDF + logistic regression\nreference (previous run)"),
]


def main() -> None:
    rows = [json.loads((ROOT / "results" / f"{method}_test.json").read_text())
            for method, _ in METHODS]
    n = rows[0]["n"]
    if any(row["n"] != n or row["split"] != "test" for row in rows):
        raise ValueError("all results must use the same full test split")
    x, width = np.arange(len(rows)), 0.34
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.8)
    for offset, key, color in ((-width / 2, "hit@1", "#1F5C70"),
                               (width / 2, "hit@3", "#8FB8C6")):
        values = np.array([row[key] for row in rows])
        lower = np.array([row[f"{key}_ci95"][0] for row in rows])
        upper = np.array([row[f"{key}_ci95"][1] for row in rows])
        bars = ax.bar(x + offset, values, width, color=color,
                      yerr=[values - lower, upper - values], capsize=5,
                      label=key.replace("hit", "Hit"),
                      error_kw={"elinewidth": 1.2, "ecolor": "#333333"})
        for bar, value, top in zip(bars, values, upper, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, top + 1.7, f"{value:.2f}",
                    ha="center", fontsize=13, color="#1A1A1A")
    ax.axvline(1.5, color="#AAB2BA", linestyle=":", linewidth=1)
    ax.set_xticks(x, [label for _, label in METHODS], fontsize=12)
    ax.set_ylim(0, 88)
    ax.set_yticks(np.arange(0, 81, 20))
    ax.set_ylabel("% of test cases", fontsize=12)
    ax.set_title("Qwen3-8B: effect of one LoRA epoch", fontsize=18, loc="left", pad=37)
    ax.text(0, 1.025, f"RuMedTop3 test, n = {n}. Error bars: 95% bootstrap confidence intervals.",
            transform=ax.transAxes, fontsize=11, color="#555555")
    ax.legend(loc="upper left", frameon=False, fontsize=12, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, 0.044,
             "Qwen3 base and LoRA: same 4-bit model, prompt and 105-label log-likelihood scoring.",
             fontsize=10, color="#555555")
    fig.text(0.02, 0.018,
             "TF-IDF is a previously reproduced reference; it was not rerun in this evaluation.",
             fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    output = ROOT / "docs/results_qwen3_test.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
