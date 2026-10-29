import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
with (ROOT / "results.json").open(encoding="utf-8") as f:
    results = json.load(f)

names = list(results["models"])
labels = ["Logistic", "HistGradientBoosting"]
metrics = ["bACC", "AUC"]
values = np.array([[results["models"][name][metric] for metric in metrics] for name in names])

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(9, 5.6), dpi=180)
x = np.arange(len(metrics))
width = 0.34
colors = ["#1769aa", "#e07a2d"]
for i, (label, color) in enumerate(zip(labels, colors)):
    bars = ax.bar(x + (i - 0.5) * width, values[i], width, label=label, color=color)
    for bar, value in zip(bars, values[i]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.4f}",
                ha="center", va="bottom", fontsize=9)

ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1.2, label="0.5 reference")
ax.set_xticks(x, metrics)
ax.set_ylim(0.45, 0.60)
ax.set_ylabel("Score")
ax.set_title("CogState baseline: subject-grouped cross-validation")
ax.text(0.01, 0.02,
        f"n={results['n_samples']:,} windows, {results['n_subjects']} subjects | "
        "5-fold GroupKFold by subject; not official held-out test performance",
        transform=ax.transAxes, fontsize=8.5, color="#555555", va="bottom")
ax.legend(frameon=True, loc="upper right")
fig.tight_layout()
fig.savefig(ROOT / "results_comparison.png", bbox_inches="tight")
