"""E8：本人方向能否分组？分组已知时的上界是多少？

E7 结论：按无标签信号属性分组，各组逐被试 AUC 几乎相同（0.638–0.686），
可靠性预测 r=+0.055，弃权/加权都无增益 → 表现差异不可由"可观测属性"识别。

本脚本追问一个更根本的问题：57 个"本人方向"本身是否成簇？
  * 若是（例如只有 3–4 种方向），那么"分组"就有物理意义，剩下的是无标签分组问题；
  * 若两两近似正交（cos≈0），则"分组"这个概念不成立，每个被试就是一个独立的模式。

做法：
  1. 逐被试算本人方向 d_s（z 化特征的 task−relax 均值差，单位化）；
  2. 报两两 |cos| 的分布（中位/90 分位）与"最大特征值占比"（方向集合的低秩程度）；
  3. 对 d_s 做 KMeans（k=2..6，先把每个 d_s 的符号对齐到全局方向以便聚"轴"），报轮廓系数；
  4. **分组已知的上界**：在每个簇内做留一被试 LR（只用同簇其他被试训练），
     把分数按簇内 z 化后池化，报 AUC/bACC；与全局 LR 的 OOF 对比。

运行：py -3.11 test/10_projects/eeg_only_20261102/direction_groups.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
T0 = time.perf_counter()


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def report(y, s, g, tag):
    yh = (s >= 0).astype(int)
    per = [balanced_accuracy_score(y[g == q], yh[g == q]) for q in np.unique(g)]
    out = {"AUC": float(roc_auc_score(y, s)), "bACC": float(balanced_accuracy_score(y, yh)),
           "F1_macro": float(f1_score(y, yh, average="macro")),
           "subject_macro_bACC": float(np.mean(per)),
           "n_ge_0.9": int(np.sum(np.array(per) >= 0.9)), "n_subj": len(per)}
    print(f"[{tag:22s}] AUC={out['AUC']:.4f} bACC={out['bACC']:.4f} F1m={out['F1_macro']:.4f} "
          f"macro={out['subject_macro_bACC']:.4f} n>=.9={out['n_ge_0.9']}", flush=True)
    return out


def main():
    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F = np.nan_to_num(D["F"].astype(float))
    y, g = D["y"].astype(int), D["g"]
    subs = sorted(np.unique(g))
    print(f"[data] windows={len(y)} subs={len(subs)} dim={F.shape[1]} {el()}", flush=True)

    # 本人方向（z 化后 task−relax）
    dirs = np.zeros((len(subs), F.shape[1]))
    for i, q in enumerate(subs):
        m = g == q
        Z = F[m]
        sd = Z.std(0)
        Z = (Z - Z.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
        d = Z[y[m] == 1].mean(0) - Z[y[m] == 0].mean(0)
        dirs[i] = d / max(np.linalg.norm(d), 1e-12)

    C = np.abs(dirs @ dirs.T)
    iu = np.triu_indices(len(subs), 1)
    pair = C[iu]
    ev = np.linalg.svd(dirs, compute_uv=False)
    frac = float(ev[0] ** 2 / (ev ** 2).sum())
    print(f"[directions] |cos| 中位={np.median(pair):.3f} 90分位={np.percentile(pair, 90):.3f} "
          f"最大={pair.max():.3f} | 第一奇异值能量占比={frac:.3f}", flush=True)

    glob = dirs.mean(0)
    glob /= max(np.linalg.norm(glob), 1e-12)
    for i in range(len(subs)):
        if dirs[i] @ glob < 0:
            dirs[i] = -dirs[i]

    # 全局 LR 的 OOF（基线）
    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        folds[te] = k
    s0 = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        sc = StandardScaler().fit(F[tr])
        m = LogisticRegression(C=.01, max_iter=2000).fit(sc.transform(F[tr]), y[tr])
        s0[te] = m.decision_function(sc.transform(F[te]))
    out = {"n_subjects": len(subs), "pair_abs_cos": {"median": float(np.median(pair)),
           "p90": float(np.percentile(pair, 90)), "max": float(pair.max())},
           "svd_energy_top1": frac, "baseline_oof": report(y, s0, g, "global_oof")}

    # KMeans 分组 + 组内留一被试
    out["kmeans"] = {}
    for kk in (2, 3, 4, 5, 6):
        km = KMeans(kk, n_init=20, random_state=0).fit(dirs)
        lab = km.labels_
        sil = float(silhouette_score(dirs, lab)) if kk > 1 else None
        s = np.full(len(y), np.nan)
        sizes = []
        for c in range(kk):
            mem = [subs[i] for i in range(len(subs)) if lab[i] == c]
            sizes.append(len(mem))
            if len(mem) < 2:
                for q in mem:
                    m = g == q
                    s[m] = s0[m]
                continue
            for q in mem:
                trm = np.zeros(len(y), bool)
                for qq in mem:
                    if qq != q:
                        trm |= g == qq
                tem = g == q
                sc = StandardScaler().fit(F[trm])
                m = LogisticRegression(C=.01, max_iter=2000).fit(sc.transform(F[trm]), y[trm])
                sc_ = m.decision_function(sc.transform(F[tem]))
                s[tem] = (sc_ - sc_.mean()) / max(sc_.std(), 1e-6)
        r = report(y, s, g, f"within_cluster k={kk}")
        r["sizes"] = sizes
        r["silhouette"] = sil
        out["kmeans"][str(kk)] = r

    out["seconds"] = time.perf_counter() - T0
    (HERE / "direction_groups.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {out['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
