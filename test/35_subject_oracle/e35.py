"""E35：57人逐被试测试段 oracle（过拟合上限，不是通用折外结果）。

在 E31b 的 1152 配置×85 因果阈值及18个原组配方中，直接使用每个目标被试的
测试 bACC 选择其最佳配方。该结果只回答“现有候选在允许测试段过拟合时能到多少”，
不可用于宣称对新被试或未见过测试段的泛化能力。

运行：.venv\Scripts\python.exe test/35_subject_oracle/e35.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "26_merge2"))
from e26lib import THR2  # noqa: E402


def main() -> None:
    cache = np.load(ROOT / "31_integrate" / "e31b_ev.npz")
    ev, orig = cache["EV"], cache["ORIG"]
    subs = np.load(ROOT / "28_block_invariant" / "e28_ev.npz")["subs"].tolist()
    params = json.loads((ROOT / "31_integrate" / "e31b.json").read_text(encoding="utf-8"))["params"]
    pools = [
        f"b|pool|w{w}|a{a:g}|t{t}"
        for w in params["WGRID"] for t in params["TAUS"] for a in params["ALPHAS"]
    ]
    names = pools + [f"z[{p};{c}]" for p in pools for c in params["comps"]]
    assert ev.shape == (57, len(names), len(THR2), 3)
    groups = json.loads((ROOT / "26_merge2" / "round2.json").read_text(encoding="utf-8"))["groups"]
    assert orig.shape[:2] == (57, len(groups))

    rows = []
    per_subject = {}
    for i, q in enumerate(subs):
        bacc = ev[i, ..., 0]
        flat = int(np.nanargmax(bacc))
        c, t = divmod(flat, len(THR2))
        best = ev[i, c, t].copy()
        choice = {"source": "E31b", "config": names[c], "threshold": THR2[t][0]}
        g = int(np.nanargmax(orig[i, :, 0]))
        if orig[i, g, 0] > best[0]:
            best = orig[i, g].copy()
            choice = {"source": "round2", "group": g, "members": groups[g]["members"],
                      "recipe": groups[g]["rec"], "threshold": "fixed"}
        rows.append(best)
        per_subject[q] = {**choice, "bacc": float(best[0]), "auc": float(best[1]), "f1": float(best[2])}

    values = np.asarray(rows)
    result = {
        "protocol": "逐人使用本人测试标签选择配置；仅为过拟合/oracle上限，不是通用或折外结果",
        "bacc": float(values[:, 0].mean()),
        "auc": float(values[:, 1].mean()),
        "f1": float(values[:, 2].mean()),
        "n90": int((values[:, 0] >= 0.9).sum()),
        "min_bacc": float(values[:, 0].min()),
        "per_subject": per_subject,
    }
    (HERE / "e35.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"bACC={result['bacc']:.6f} AUC={result['auc']:.6f} F1={result['f1']:.6f} "
          f"n90={result['n90']}/57 min={result['min_bacc']:.6f}")
    print("source counts:", {s: sum(v["source"] == s for v in per_subject.values()) for s in ("E31b", "round2")})


if __name__ == "__main__":
    main()
