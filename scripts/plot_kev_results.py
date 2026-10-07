"""Generate quality and reliability figures from verified local Kev evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    kev = json.loads((ROOT / "results/kev_test.json").read_text())
    methods = [("Kev-0.8B\nzero-shot, Russian labels", kev["metrics"]),
               ("Qwen3-8B\nzero-shot, previous run",
                json.loads((ROOT / "results/local_zero_shot_test.json").read_text())),
               ("Qwen3-8B + LoRA\none epoch, previous run",
                json.loads((ROOT / "results/local_lora_test.json").read_text())),
               ("DeepSeek\nzero-shot, previous run",
                json.loads((ROOT / "results/zero_shot_test.json").read_text())),
               ("DeepSeek\n15 fixed cases, previous run",
                json.loads((ROOT / "results/few_shot_test.json").read_text())),
               ("DeepSeek + RAG\n15 nearest cases, previous run",
                json.loads((ROOT / "results/rag_test.json").read_text())),
               ("TF-IDF + LR\npaired rerun", kev["tfidf_control"])]
    if any(row["n"] != 822 for _, row in methods):
        raise ValueError("expected the same full test split")
    x, width = np.arange(len(methods)), .34
    fig, ax = plt.subplots(figsize=(17, 7), dpi=120)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E5E7EB")
    for offset, key, color in ((-width / 2, "hit@1", "#1F5C70"),
                               (width / 2, "hit@3", "#8FB8C6")):
        values = np.array([r[key] for _, r in methods])
        low = np.array([r[f"{key}_ci95"][0] for _, r in methods])
        high = np.array([r[f"{key}_ci95"][1] for _, r in methods])
        bars = ax.bar(x + offset, values, width, color=color, label=key.replace("hit", "Hit"),
                      yerr=[values - low, high - values], capsize=4)
        for bar, value, top in zip(bars, values, high, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, top + 1.2, f"{value:.2f}",
                    ha="center", fontsize=12)
    ax.set_xticks(x, [name for name, _ in methods], fontsize=10)
    ax.set_ylim(0, 90)
    ax.set_ylabel("% of test cases")
    ax.set_title("RuMedTop3: local typed decisions vs earlier approaches", loc="left", pad=32)
    ax.text(0, 1.025,
            "Test n = 822. Error bars: 95% bootstrap CI. Same pinned dataset, different protocols.",
            transform=ax.transAxes, fontsize=10)
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .045,
             "Kev: pretrained English model, Russian complaints and labels; no RuMed fine-tuning.",
             fontsize=9)
    fig.text(.02, .018,
             "Kev receives descriptions; Qwen scores code tokens; DeepSeek generates JSON; "
             "TF-IDF learns from train.",
             fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1))
    fig.savefig(ROOT / "docs/results_kev_test.png")
    plt.close(fig)

    calibration = kev["calibration"]
    bins = [b for b in calibration["bins"] if b["n"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="perfect calibration")
    ax.scatter([b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins],
               s=[25 + b["n"] / 2 for b in bins], color="#1F5C70", label="occupied bins")
    for b in bins:
        ax.text(b["mean_confidence"], b["accuracy"] + .035, f"n={b['n']}",
                ha="center", fontsize=9)
    ax.set(xlim=(0, 1), ylim=(0, 1.1), yticks=np.arange(0, 1.01, .2),
           xlabel="Mean top-1 probability",
           ylabel="Observed top-1 accuracy",
           title="Kev reliability on Russian complaints")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    path = ROOT / "results/kev_test_predictions.jsonl"
    timings = [json.loads(line)["kev_seconds"] * 1000 for line in path.read_text().splitlines()]
    ax.hist(timings, bins=25, color="#8FB8C6", edgecolor="white")
    ax.set(xlabel="Milliseconds per case", ylabel="Number of cases",
           title="Local inference, batch 1, warm model")
    ax.axvline(kev["latency"]["median_ms"], color="#1F5C70", label="median")
    ax.axvline(kev["latency"]["p95_ms"], color="#C06E3A", linestyle="--", label="p95")
    ax.legend(frameon=False)
    fig.text(.02, .02,
             "M2 Pro, 16 GB; MLX bf16 backbone, fp32 head. "
             f"ECE(10 bins)={calibration['ece_10_equal_width_bins']:.3f}. "
             "Shipped temperature; calibration on RuMed was not fitted.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(ROOT / "docs/results_kev_reliability.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
