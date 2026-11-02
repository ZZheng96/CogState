"""E6：天花板审计——四种口径测同一条上限，并检查是否被少数坏记录拖累。

四种口径（同一批 57 人、16 067 窗、真实顺序，全部不使用次序/位置）：
  1. 跨被试 LR（严格可部署）
  2. oracle polarity：若某被试本人的符号与模型相反就翻转（= 只给每个人 1 bit 的符号，不可部署）
  3. oracle direction：用本人全部 task/relax 窗的均值差当方向（不可部署，线性上限）
  4. 剔除三个已知质量差记录（008 饱和 18%、009 工频 29×、034 坏段 56%）后重算 1 与 3

目的：给出"0.95 还差多少"的最终量化，并排除"只是坏数据拖累"的解释。

运行：py -3.11 test/10_projects/eeg_only_20261102/ceiling_audit.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
T0 = time.perf_counter()
BAD = ("008", "009", "034")


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def stats(y, s, g, tag):
    yh = (s >= 0).astype(int)
    per = {q: roc_auc_score(y[g == q], s[g == q]) for q in np.unique(g) if y[g == q].min() != y[g == q].max()}
    yhb = (s >= 0).astype(int)
    perb = {q: balanced_accuracy_score(y[g == q], yhb[g == q]) for q in np.unique(g)}
    out = {"AUC": float(roc_auc_score(y, s)), "bACC": float(balanced_accuracy_score(y, yhb)),
           "F1_macro": float(f1_score(y, yhb, average="macro")),
           "subject_macro_bACC": float(np.mean(list(perb.values()))),
           "n_ge_0.9": int(np.sum(np.array(list(perb.values())) >= 0.9)),
           "subject_auc_median": float(np.median(list(per.values()))),
           "subject_auc_min": float(np.min(list(per.values()))),
           "subject_auc_max": float(np.max(list(per.values()))),
           "n_subject_auc_below_0.5": int(np.sum(np.array(list(per.values())) < 0.5)),
           "n_subj": len(per)}
    print(f"[{tag:22s}] AUC={out['AUC']:.4f} bACC={out['bACC']:.4f} F1m={out['F1_macro']:.4f} "
          f"macro={out['subject_macro_bACC']:.4f} n>=.9={out['n_ge_0.9']} "
          f"| subjAUC med={out['subject_auc_median']:.3f} [{out['subject_auc_min']:.3f},{out['subject_auc_max']:.3f}] "
          f"flip={out['n_subject_auc_below_0.5']}/{out['n_subj']}", flush=True)
    return out


def main():
    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F, y, g, t = D["F"].astype(float), D["y"].astype(int), D["g"], D["t"]
    Z = np.nan_to_num(F)
    print(f"[data] windows={len(y)} subs={len(np.unique(g))} dim={Z.shape[1]} {el()}", flush=True)
    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        folds[te] = k

    # 1 跨被试 LR（严格）
    s = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        sc = StandardScaler().fit(Z[tr])
        m = LogisticRegression(C=.01, max_iter=2000).fit(sc.transform(Z[tr]), y[tr])
        s[te] = m.decision_function(sc.transform(Z[te]))
    res = {"windows": int(len(y)), "subjects": int(len(np.unique(g))), "results": {}}
    res["results"]["cross_subject_lr"] = stats(y, s, g, "cross_subject_lr")

    # 2 oracle polarity：只给每人 1 bit 符号
    s2 = s.copy()
    for q in np.unique(g):
        m = g == q
        if roc_auc_score(y[m], s[m]) < 0.5:
            s2[m] = -s[m]
    res["results"]["oracle_polarity"] = stats(y, s2, g, "oracle_polarity")

    # 3 oracle direction：本人均值差方向（线性上限）
    s3 = np.full(len(y), np.nan)
    for q in np.unique(g):
        m = g == q
        Zs = Z[m]
        sd = Zs.std(0)
        Zs = (Zs - Zs.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
        d = Zs[y[m] == 1].mean(0) - Zs[y[m] == 0].mean(0)
        n = np.linalg.norm(d)
        s3[m] = Zs @ (d / max(n, 1e-12))
    res["results"]["oracle_direction"] = stats(y, s3, g, "oracle_direction")

    # 4 剔除坏记录
    keep = ~np.isin(g, BAD)
    res["results"]["cross_subject_lr_no_bad"] = stats(y[keep], s[keep], g[keep], "lr_no_bad")
    res["results"]["oracle_direction_no_bad"] = stats(y[keep], s3[keep], g[keep], "oracle_dir_no_bad")

    res["seconds"] = time.perf_counter() - T0
    (HERE / "ceiling_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {res['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
