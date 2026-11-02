"""E18b：本人标注为什么在留一块下失效？——块偏移主导，先做特征级因果标准化再训练。

E18 的发现：逐被试留一块（LOBO）本人 LR 的 AUC 常常只有 0.1–0.5，即**块间几乎不迁移**。
这与 D1（留一块 .33）一致：模型学到的是"每块自己的偏移"，而不是状态方向。
本脚本在训练前先对**特征**做因果标准化（过去 W 帧的均值/标准差），把块级偏移压掉，
再看本人 LOBO 能否恢复；对比：
  raw    原始 878 维
  cstd   特征级因果标准化（W=150 帧 = 300 s）
  glb    全局模型在同一特征上的表现（同一标准化）

运行：py -3.11 test/11_individual_20261102/feat_cstd.py [--subjects 018,036,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLID = HERE / "sliding"
SC = HERE / "scores"


def running_stats(X, w=150):
    """因果滑动均值/标准差（含当前帧），O(n·d)。"""
    n = len(X)
    c1 = np.cumsum(np.vstack([np.zeros(X.shape[1]), X]), axis=0)
    c2 = np.cumsum(np.vstack([np.zeros(X.shape[1]), X ** 2]), axis=0)
    idx = np.arange(1, n + 1)
    lo = np.maximum(0, idx - w)
    cnt = (idx - lo)[:, None]
    m = (c1[idx] - c1[lo]) / cnt
    v = (c2[idx] - c2[lo]) / cnt - m ** 2
    return m, np.sqrt(np.maximum(v, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="018,036,054,044,029,037,024,004,048,056")
    a = ap.parse_args()
    subs = a.subjects.split(",")
    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F = np.nan_to_num(D["F"].astype(np.float32))
    y_all, g = D["y"].astype(int), D["g"]
    from sklearn.model_selection import StratifiedGroupKFold
    folds = np.empty(len(y_all), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                .split(np.zeros(len(y_all)), y_all, g)):
        folds[te] = k
    fold_of = {q: int(folds[g == q][0]) for q in sorted(np.unique(g))}

    out = {}
    print(f"{'sub':4s} {'rawAUC':>7s} {'cstdAUC':>8s} {'glb_raw':>8s} {'glb_cstd':>9s} "
          f"{'cstd_bACC':>10s}")
    for q in subs:
        p = SLID / f"sub-{q}.npz"
        if not p.exists():
            print(f"[skip] {q}")
            continue
        S = np.load(p)
        X = np.nan_to_num(S["F"].astype(np.float32))
        y = S["y"].astype(int)
        blk = S["blk"].astype(int)
        o = np.argsort(S["t"])
        X, y, blk = X[o], y[o], blk[o]
        m, sd = running_stats(X)
        Xc = (X - m) / sd

        def lobo(Z):
            s = np.full(len(y), np.nan)
            for b in np.unique(blk):
                te = blk == b
                tr = ~te
                if len(np.unique(y[tr])) < 2:
                    continue
                sc = StandardScaler().fit(Z[tr])
                mdl = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Z[tr]), y[tr])
                s[te] = mdl.decision_function(sc.transform(Z[te]))
            ok = np.isfinite(s)
            return roc_auc_score(y[ok], s[ok]), balanced_accuracy_score(y[ok], (s[ok] >= 0).astype(int))

        r_raw = lobo(X)
        r_cstd = lobo(Xc)
        tr = np.flatnonzero(folds != fold_of[q])
        sc = StandardScaler().fit(F[tr])
        mdl = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(F[tr]), y_all[tr])
        g_auc = float(roc_auc_score(y, mdl.decision_function(sc.transform(X))))
        out[q] = {"raw_lobo_auc": r_raw[0], "cstd_lobo_auc": r_cstd[0],
                  "cstd_lobo_bacc": r_cstd[1], "global_raw": g_auc}
        print(f"{q:4s} {r_raw[0]:7.3f} {r_cstd[0]:8.3f} {g_auc:8.3f} "
              f"{r_cstd[1]:10.3f}", flush=True)
    (HERE / "feat_cstd.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print("[done] feat_cstd.json")


if __name__ == "__main__":
    main()
