from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train import classification_metrics  # noqa: E402

results = ROOT / "test" / "results"
eeg = pd.read_csv(results / "oof_predictions.csv")
protocol = pd.read_csv(results / "protocol_template_oof.csv")
eeg["second"] = np.rint(eeg["endpoint"] / 250).astype(int)
data = eeg.merge(
    protocol,
    on=["record", "second", "y"],
    how="inner",
    suffixes=("_eeg", "_protocol"),
    validate="one_to_one",
)
y = data["y"].to_numpy()
e = data["score_task_eeg"].to_numpy()
p = data["score_task_protocol"].to_numpy()
rows = []

def add(method, weight, scores):
    rows.append({
        "method": method,
        "eeg_weight": weight,
        **classification_metrics(y, scores),
    })

add("protocol", 0.0, p)
add("eeg", 1.0, e)
for weight in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
    add("linear", weight, (1 - weight) * p + weight * e)
    add("logit", weight, expit(logit(np.clip(p, 1e-4, 1 - 1e-4)) + weight * logit(np.clip(e, 1e-4, 1 - 1e-4))))
    uncertainty = 4 * p * (1 - p)
    add("boundary_only", weight, np.clip(p + weight * uncertainty * (e - 0.5), 0, 1))

comparison = pd.DataFrame(rows).sort_values(["bacc", "auc"], ascending=False)
comparison.to_csv(results / "protocol_fusion_comparison.csv", index=False)
print("Merged labeled valid windows:", len(data))
print(comparison[["method", "eeg_weight", "bacc", "auc", "relax_recall", "task_recall"]].to_string(index=False))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

selected = pd.DataFrame([
    comparison.loc[comparison["method"] == "eeg"].iloc[0],
    comparison.loc[comparison["method"] == "protocol"].iloc[0],
    comparison.loc[(comparison["method"] == "boundary_only") & np.isclose(comparison["eeg_weight"], 0.1)].iloc[0],
])
labels = ["Causal EEG only\n(formal result)", "Recording-time template\n(shortcut audit)", "Time template + EEG\n(shortcut audit)"]
x = np.arange(3)
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax.bar(x - 0.18, selected["bacc"], width=0.36, label="bACC @ 0.5", color="#2878B5")
ax.bar(x + 0.18, selected["auc"], width=0.36, label="AUC", color="#F58518")
for xpos, values in zip(x, selected[["bacc", "auc"]].to_numpy()):
    ax.text(xpos - 0.18, values[0] + 0.008, f"{values[0]:.4f}", ha="center", fontsize=9)
    ax.text(xpos + 0.18, values[1] + 0.008, f"{values[1]:.4f}", ha="center", fontsize=9)
ax.axhline(0.9, color="#0E9F6E", linestyle="--", linewidth=1.2, label="Target 0.9")
ax.set_xticks(x, labels)
ax.set_ylim(0.5, 1.065)
ax.set_ylabel("Grouped OOF score")
ax.set_title("Protocol shortcut audit — not a valid pure-EEG optimization result", color="#B42318")
ax.grid(axis="y", alpha=0.2)
ax.legend(loc="lower right")
fig.savefig(results / "protocol_fusion_comparison.png", dpi=180)
plt.close(fig)
