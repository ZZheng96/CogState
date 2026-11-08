"""E76 部署门控：用全部 57 人的 E74 折外组件分数拟合同一 Logistic 门控，导出 deploy_gate.json。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from e76 import features, load

HERE = Path(__file__).resolve().parent


def main():
    data = load(); assert len(data) == 57
    X = np.concatenate([features(d["lr_scaled"], d["mlp_scaled"]) for d in data.values()])
    y = np.concatenate([d["y"] for d in data.values()])
    gate = LogisticRegression(C=0.1, class_weight="balanced", max_iter=300, solver="liblinear", random_state=0).fit(X, y)
    out = {"source": "E74 out-of-fold lr_scaled/mlp_scaled, 57 subjects", "window": 120,
           "features": ["lr", "mlp", "abs_diff", "lr_mean", "mlp_mean", "diff_mean", "lr_std", "mlp_std", "diff_std"],
           "coef": gate.coef_[0].tolist(), "intercept": float(gate.intercept_[0]), "shrink": 0.5,
           "score": "shrink*gate + (1-shrink)*(lr+mlp) -> EMA1200 + HMM300"}
    (HERE / "deploy_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
