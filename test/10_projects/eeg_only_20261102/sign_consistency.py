"""E2：是否存在"方向一致"的单特征？——跨被试是否为每个特征都发生符号翻转。

背景：跨被试 AUC 只有 0.645，主要原因是约 1/3 被试的判别方向相反；而无标签数据里无法定符号
（label switching）。本诊断回答一个更具体的问题：**有没有某些特征，其 task-relax 效应的符号在
57 人上高度一致？** 若有，就能绕开"定符号"这一步，直接用固定符号做跨被试判别。

做法：
  * 逐被试、逐特征计算 Cohen's d = (mean_task − mean_relax) / pooled_sd；
  * 符号一致率 = 该特征 d 的符号与全体中位符号一致的被试比例（0.5 = 随机）；
  * 单特征池化 AUC（用中位符号定向）。

输出：sign_consistency.json / .log
运行：py -3.11 test/10_projects/eeg_only_20261102/sign_consistency.py
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DB = ROOT / "database" / "windows"
T0 = time.perf_counter()


def load():
    F, E, Y, G = [], [], [], []
    for nm, lab in (("relax", 0), ("task", 1)):
        for p in sorted(DB.joinpath(nm).glob("sub-*.npz")):
            d = np.load(p)
            F.append(d["f"]); E.append(d["e"])
            Y.append(np.full(len(d["f"]), lab)); G.append(np.array([p.stem.split("-")[1]] * len(d["f"])))
    return np.concatenate(F), np.concatenate(E), np.concatenate(Y).astype(int), np.concatenate(G)


def main():
    F, E, y, g = load()
    Z = np.hstack([F, E])
    manifest = json.loads((ROOT / "database" / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["feature_names"] + manifest["emg_feature_names"]
    subs = np.unique(g)
    print(f"[data] windows={len(y)} subs={len(subs)} dim={Z.shape[1]}", flush=True)

    D = np.zeros((len(subs), Z.shape[1]))
    for i, s in enumerate(subs):
        q = g == s
        a, b = Z[q][y[q] == 1], Z[q][y[q] == 0]
        sd = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-12
        D[i] = (a.mean(0) - b.mean(0)) / sd
    med = np.median(D, 0)
    cons = (np.sign(D) == np.sign(med)).mean(0)
    auc = np.array([roc_auc_score(y, Z[:, j]) for j in range(Z.shape[1])])
    eff = np.abs(auc - 0.5)

    order = np.argsort(-eff)[:15]
    print("[top] |AUC-.5| 最大的 15 个单特征：")
    for j in order:
        print(f"  {names[j][:44]:44s} AUC={auc[j]:.3f} cons={cons[j]:.2f} medD={med[j]:+.2f}", flush=True)
    hi = eff > 0.08
    res = {"n_windows": int(len(y)), "n_subjects": int(len(subs)), "dim": int(Z.shape[1]),
           "top_by_effect": [{"name": names[j], "AUC": float(auc[j]), "consistency": float(cons[j]),
                              "median_d": float(med[j])} for j in order],
           "max_consistency_eff_gt_0.08": float(cons[hi].max()) if hi.any() else None,
           "n_eff_gt_0.08": int(hi.sum()),
           "mean_consistency_all": float(cons.mean()),
           "best_single": None, "seconds": 0.0}
    m = eff > 0.06
    if m.any():
        j = int(np.argmax(np.where(m, cons, -1)))
        res["best_single"] = {"name": names[j], "AUC": float(auc[j]), "consistency": float(cons[j])}
        print(f"[best] 高效应特征里符号最一致：{names[j]} AUC={auc[j]:.3f} cons={cons[j]:.3f}", flush=True)
    print(f"[summary] mean_consistency={cons.mean():.3f} max_consistency(|AUC-.5|>0.08)="
          f"{res['max_consistency_eff_gt_0.08']:.3f} n={res['n_eff_gt_0.08']}", flush=True)
    res["seconds"] = time.perf_counter() - T0
    (HERE / "sign_consistency.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {res['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
