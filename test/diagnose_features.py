from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_decoder import model_scores  # noqa: E402

cache = np.load(ROOT / "test/cache/features_w4_s1_welch2_fs250.npz", allow_pickle=False)
X, y, pure, groups = (cache[name] for name in ["X", "y", "pure", "groups"])
rows = []
for record in np.unique(groups):
    mask = groups == record
    xr, yr, pr = X[mask], y[mask], pure[mask]
    blocks = np.concatenate([[0], np.cumsum(yr[1:] != yr[:-1])])
    scores = np.full(len(yr), np.nan)
    for block in np.unique(blocks):
        test = blocks == block
        train = (blocks != block) & pr
        if len(np.unique(yr[train])) < 2:
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)),
        ])
        model.fit(xr[train], yr[train])
        scores[test] = model_scores(model, xr[test])
    valid = np.isfinite(scores)
    rows.append({
        "record": record,
        "bacc": balanced_accuracy_score(yr[valid], scores[valid] >= 0.5),
        "auc": roc_auc_score(yr[valid], scores[valid]),
    })
    print(f"{record}: bACC={rows[-1]['bacc']:.4f}, AUC={rows[-1]['auc']:.4f}", flush=True)
print("SUMMARY", {
    "median_bacc": float(np.median([row["bacc"] for row in rows])),
    "median_auc": float(np.median([row["auc"] for row in rows])),
    "mean_bacc": float(np.mean([row["bacc"] for row in rows])),
    "mean_auc": float(np.mean([row["auc"] for row in rows])),
})
