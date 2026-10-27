from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_decoder import model_scores  # noqa: E402
from train import classification_metrics, make_folds  # noqa: E402

cache = np.load(ROOT / "test/cache/features_w4_s1_welch2_fs250.npz", allow_pickle=False)
X, y, pure, groups = (cache[name] for name in ["X", "y", "pure", "groups"])
folds = make_folds(groups, 6, 42)
subsets = {
    "all_37": np.arange(37),
    "absolute_and_relative_bands": np.arange(26),
    "relative_bands": np.arange(13, 26),
    "relative_and_shape_no_amplitude": np.r_[13:26, 29:37],
    "spectral_shape_only": np.r_[13:26, 32:37],
    "time_domain_only": np.arange(26, 32),
    "ratios_and_shape": np.arange(29, 37),
}
rows = []
for name, columns in subsets.items():
    scores = np.full(len(y), np.nan)
    for test_records in folds:
        test = np.isin(groups, test_records)
        train = (~np.isin(groups, test_records)) & pure
        model = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1, class_weight="balanced", max_iter=1000)),
        ])
        model.fit(X[train][:, columns], y[train])
        scores[test] = model_scores(model, X[test][:, columns])
    metrics = classification_metrics(y, scores)
    rows.append({"subset": name, "features": len(columns), **metrics})
    print(rows[-1], flush=True)
pd.DataFrame(rows).sort_values("auc", ascending=False).to_csv(ROOT / "test/results/feature_ablation.csv", index=False)
