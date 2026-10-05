"""Plot test Hit@1 / Hit@3 with 95% bootstrap CIs from results/*_test.json.

Usage: uv run --with matplotlib python scripts/plot_results.py
Writes docs/results_test.png (1200x627, LinkedIn link-image size).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("zero_shot", "LLM\nzero-shot"),
    ("few_shot", "LLM\n15 fixed examples"),
    ("rag", "LLM + RAG\n15 nearest cases"),
    ("tfidf", "TF-IDF +\nlogistic regression"),
]
PAPER = {"hit@1": 49.76, "hit@3": 72.75}  # RuMedBench feature-based baseline, arXiv:2201.06499


def main() -> None:
    rows = [json.loads((ROOT / "results" / f"{m}_test.json").read_text()) for m, _ in METHODS]
    n = rows[0]["n"]
    x = np.arange(len(METHODS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(12, 6.27), dpi=100)
    colors = {"hit@1": "#1F5C70", "hit@3": "#8FB8C6"}
    for off, key, label in ((-w / 2, "hit@1", "Hit@1"), (w / 2, "hit@3", "Hit@3")):
        vals = np.array([r[key] for r in rows])
        lo = vals - np.array([r[f"{key}_ci95"][0] for r in rows])
        hi = np.array([r[f"{key}_ci95"][1] for r in rows]) - vals
        bars = ax.bar(x + off, vals, w, yerr=[lo, hi], capsize=4, color=colors[key], label=label,
                      error_kw={"elinewidth": 1, "ecolor": "#333333"})
        for b, v in zip(bars, vals, strict=True):
            # round half up, so 52.55 reads 52.6 as in the text
            ax.text(b.get_x() + b.get_width() / 2, v + hi.max() + 1.2, f"{round(v + 1e-9, 1):.1f}",
                    ha="center", fontsize=12, color="#1A1A1A")
    for key, style in (("hit@1", "--"), ("hit@3", ":")):
        ax.axhline(PAPER[key], color="#B03A2E", lw=1.2, ls=style,
                   label=f"Paper baseline {key.capitalize()} {PAPER[key]:.1f}")
    ax.set_xticks(x, [label for _, label in METHODS], fontsize=12)
    ax.set_ylim(0, 92)
    ax.set_ylabel("% of test cases", fontsize=12)
    ax.set_title(f"RuMedTop3: ICD-10 code from Russian patient complaints (test, n = {n})",
                 fontsize=15, loc="left", pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=11, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "LLM: deepseek-flash, temperature 0. Error bars: 95% bootstrap CI. "
             "github.com/denis-samatov/rumed-icd-llm", fontsize=9.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / "docs" / "results_test.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
