"""E76 部署门控：用嵌套 LOSO 57 个外折的折外组件分数拟合同一 Logistic 门控，导出 deploy_gate.json。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from e76 import features  # 与 nested_loso.gate_features 相同

HERE = Path(__file__).resolve().parent
NESTED = HERE / "nested_loso" / "cache"


def main():
    files = sorted(NESTED.glob("fold_*.npz")); assert len(files) == 57
    data = [np.load(f) for f in files]
    X = np.concatenate([features(d["lr_scaled"], d["mlp_scaled"]) for d in data])
    y = np.concatenate([d["y"] for d in data])
    gate = LogisticRegression(C=0.1, class_weight="balanced", max_iter=300, solver="liblinear", random_state=0).fit(X, y)
    out = {"source": "E76 nested LOSO outer-fold lr_scaled/mlp_scaled (nested_loso/cache), 57 subjects", "window": 120,
           "features": ["lr", "mlp", "abs_diff", "lr_mean", "mlp_mean", "diff_mean", "lr_std", "mlp_std", "diff_std"],
           "coef": gate.coef_[0].tolist(), "intercept": float(gate.intercept_[0]), "shrink": 0.5,
           "score": "shrink*gate + (1-shrink)*(lr+mlp) -> EMA1200 + HMM300"}
    (HERE / "deploy_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
