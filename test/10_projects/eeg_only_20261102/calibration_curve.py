"""E9：每被试校准标签预算曲线（诊断，使用测试被试真标签，规则不允许）。

E7/E8 结论：个体差异真实存在（逐被试 AUC 0.33–0.85），但
  * 无法用无标签信号属性识别谁好谁坏（可靠性预测 r=+0.055）；
  * 即使把"本人方向"聚类（|cos| 中位 0.353）并给每个簇单独训练，也比全局更差（0.63 → 0.59）。
所以唯一能利用个体差异的方式是用**本人少量真标签**给方向定标。本脚本量化这条路的收益，
供"是否放宽约束"的决策使用。

做法：每个被试随机取每类 m 个窗做校准（m=1/2/5/10/20/50），
  ① 均值差方向：d = mean(task_cal) − mean(relax_cal)，再对剩余窗投影；
  ② 收缩到全局方向：d_shrunk = (1−λ)·d_global + λ·d_cal（λ 由内层选，这里固定 0.5 做参照）；
  评分只用剩余窗，池化报 AUC/bACC/F1。m=0 表示只用跨被试全局模型。

运行：py -3.11 test/10_projects/eeg_only_20261102/calibration_curve.py
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
MS = (1, 2, 5, 10, 20, 50)


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def pooled(y, s, g, tag):
    yh = (s >= 0).astype(int)
    per = [balanced_accuracy_score(y[g == q], yh[g == q]) for q in np.unique(g)]
    out = {"AUC": float(roc_auc_score(y, s)), "bACC": float(balanced_accuracy_score(y, yh)),
           "F1_macro": float(f1_score(y, yh, average="macro")),
           "subject_macro_bACC": float(np.mean(per)),
           "n_ge_0.9": int(np.sum(np.array(per) >= 0.9)), "n_subj": len(per)}
    print(f"[{tag:26s}] AUC={out['AUC']:.4f} bACC={out['bACC']:.4f} F1m={out['F1_macro']:.4f} "
          f"macro={out['subject_macro_bACC']:.4f} n>=.9={out['n_ge_0.9']}", flush=True)
    return out


def main():
    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F = np.nan_to_num(D["F"].astype(float))
    y, g = D["y"].astype(int), D["g"]
    subs = sorted(np.unique(g))
    print(f"[data] windows={len(y)} subs={len(subs)} {el()}", flush=True)

    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        folds[te] = k
    s0 = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        sc = StandardScaler().fit(F[tr])
        m = LogisticRegression(C=.01, max_iter=2000).fit(sc.transform(F[tr]), y[tr])
        s0[te] = m.decision_function(sc.transform(F[te]))
    out = {"n_subjects": len(subs), "budget": {"m=0": pooled(y, s0, g, "global m=0")}}

    rng = np.random.default_rng(0)
    # 全局方向对每个被试只算一次（用其余被试训练），供收缩使用；避免在 m 循环里重复拟合
    gdir = {}
    for q in subs:
        trg = np.flatnonzero(g != q)
        scg = StandardScaler().fit(F[trg])
        mg = LogisticRegression(C=.01, max_iter=2000).fit(scg.transform(F[trg]), y[trg])
        v = mg.coef_[0].copy(); v /= max(np.linalg.norm(v), 1e-12)
        gdir[q] = v
    print(f"[gdir] 全局方向就绪 {el()}", flush=True)

    for m in MS:
        s_dir = np.full(len(y), np.nan)
        s_shr = np.full(len(y), np.nan)
        for q in subs:
            loc = np.flatnonzero(g == q)
            Z = F[loc]
            sd = Z.std(0)
            Z = (Z - Z.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
            yy = y[loc]
            it = rng.permutation(len(loc))
            cal, ev = [], []
            for c in (0, 1):
                idx = it[yy[it] == c]
                cal.append(idx[:m])
                ev.append(idx[m:])
            cal = np.concatenate(cal)
            ev = np.concatenate(ev)
            d = Z[cal][yy[cal] == 1].mean(0) - Z[cal][yy[cal] == 0].mean(0)
            d /= max(np.linalg.norm(d), 1e-12)
            sc_ = Z[ev] @ d
            s_dir[loc[ev]] = (sc_ - sc_.mean()) / max(sc_.std(), 1e-6)
            ds = 0.5 * d + 0.5 * gdir[q]
            ds /= max(np.linalg.norm(ds), 1e-12)
            sc2 = Z[ev] @ ds
            s_shr[loc[ev]] = (sc2 - sc2.mean()) / max(sc2.std(), 1e-6)
        ev_mask = np.isfinite(s_dir)
        out["budget"][f"m={m}|direction"] = pooled(y[ev_mask], s_dir[ev_mask], g[ev_mask], f"m={m} direction")
        out["budget"][f"m={m}|shrunk0.5"] = pooled(y[ev_mask], s_shr[ev_mask], g[ev_mask], f"m={m} shrunk0.5")

    out["seconds"] = time.perf_counter() - T0
    (HERE / "calibration_curve.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {out['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
