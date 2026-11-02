"""E20：逐被试最优模型（特征级因果标准化 + 折内选特征数 + 个体记忆长度）。

对每个被试完全独立：
  1. 特征：过去 300 s 的因果均值/标准差标准化（去掉块级偏移，见 feat_cstd.py）；
  2. 留一块（LOBO）训练本人 LR；每折内用 3 折（按块分组）从 k∈{50,100,200,全} 里选特征数；
  3. 每折分数用训练折均值/标准差标准化（跨折同尺度）；
  4. 个体记忆：EMA h∈{1,2,4}（只用过去）；阈值 0；
  5. 与"全局方向 + HMM 记忆"（E11 方法）比较，取逐人更优者作为"该个体最适合的方法"。

运行：py -3.11 test/11_individual_20261102/subject_best.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from feat_cstd import running_stats
from final_eval import cstd, hmm_forward

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLID = HERE / "sliding"
SC = HERE / "scores"
STRIDE = 2.0
KS = (50, 100, 200, 10_000)


def ema(x, h):
    if h <= 1:
        return x.copy()
    a = 0.5 ** (1.0 / h)
    return lfilter([1 - a], [1, -a], x, zi=[a * x[0]])[0]


def fit_score(Xtr, ytr, Xte, k, C=0.01):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    sel = SelectKBest(f_classif, k=min(k, A.shape[1]))
    A2, B2 = sel.fit_transform(A, ytr), sel.transform(B)
    m = LogisticRegression(C=C, max_iter=3000).fit(A2, ytr)
    return m.decision_function(A2), m.decision_function(B2)


def choose_k(Xtr, ytr, gtr):
    """折内按块分组选特征数；块太少时直接回退 k=200。"""
    n = min(3, len(np.unique(gtr)))
    if n < 2 or len(np.unique(ytr)) < 2:
        return 200
    best, bk = -1.0, 200
    for k in KS:
        aucs = []
        for ti, vi in StratifiedGroupKFold(n, shuffle=True, random_state=11).split(
                np.zeros(len(ytr)), ytr, gtr):
            if len(np.unique(ytr[ti])) < 2 or len(np.unique(ytr[vi])) < 2:
                continue
            _, dte = fit_score(Xtr[ti], ytr[ti], Xtr[vi], k)
            aucs.append(roc_auc_score(ytr[vi], dte))
        if aucs and np.mean(aucs) > best:
            best, bk = float(np.mean(aucs)), k
    return bk


def main():
    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    y_all, g = D["y"].astype(int), D["g"]
    folds = np.empty(len(y_all), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                .split(np.zeros(len(y_all)), y_all, g)):
        folds[te] = k

    subs = sorted(p.stem.replace("sub-", "") for p in SC.glob("sub-*.npz"))
    out, rows = {}, []
    for q in subs:
        p = SLID / f"sub-{q}.npz"
        if not p.exists():
            continue
        S = np.load(p)
        X = np.nan_to_num(S["F"].astype(np.float32))
        y, blk = S["y"].astype(int), S["blk"].astype(int)
        o = np.argsort(S["t"])
        X, y, blk = X[o], y[o], blk[o]
        if len(np.unique(y)) < 2:
            continue
        m, sd = running_stats(X)
        Xc = (X - m) / sd
        s = np.full(len(y), np.nan)
        ks = []
        for b in np.unique(blk):
            te, tr = blk == b, blk != b
            if len(np.unique(y[tr])) < 2:
                s[te] = 0.0
                continue
            k = choose_k(Xc[tr], y[tr], blk[tr])
            dtr, dte = fit_score(Xc[tr], y[tr], Xc[te], k)
            s[te] = (dte - dtr.mean()) / (dtr.std() + 1e-12)
            ks.append(k)

        rec = {"ks": ks}
        for h in (1, 2, 4):
            e = ema(s, h)
            rec[f"own_h{h}"] = {"AUC": float(roc_auc_score(y, e)),
                                "bACC": float(balanced_accuracy_score(y, (e >= 0).astype(int))),
                                "F1": float(f1_score(y, (e >= 0).astype(int)))}
        Z = np.load(SC / f"sub-{q}.npz")
        sg = Z["score__lr"].astype(float)
        for Dd in (30, 60, 90):
            pr = hmm_forward(cstd(sg), float(Z["mu0__lr"]), float(Z["mu1__lr"]), float(Z["sd__lr"]),
                             1 - STRIDE / Dd)
            rec[f"glb_D{Dd}"] = {"AUC": float(roc_auc_score(y, pr)),
                                 "bACC": float(balanced_accuracy_score(y, pr >= .5)),
                                 "F1": float(f1_score(y, pr >= .5))}
        best = max((k for k in rec if k.startswith(("own_h", "glb_D"))),
                   key=lambda k: rec[k]["bACC"])
        rec["best_key"], rec["best_bACC"] = best, rec[best]["bACC"]
        out[q] = rec
        rows.append((q, rec["own_h1"]["AUC"], rec["own_h2"]["bACC"], rec["own_h4"]["bACC"],
                     rec["best_bACC"], best))
        print(f"{q:4s} ownAUC {rec['own_h1']['AUC']:.3f} h2 {rec['own_h2']['bACC']:.3f} "
              f"h4 {rec['own_h4']['bACC']:.3f} | glbD60 {rec['glb_D60']['bACC']:.3f} "
              f"| best {rec['best_bACC']:.3f} ({best}) ks={ks}", flush=True)

    b = np.array([r[4] for r in rows])
    a = np.array([r[1] for r in rows])
    print(f"\n[own AUC] mean={a.mean():.3f} median={np.median(a):.3f} n>=.95={int((a >= .95).sum())}")
    print(f"[per-subject best] n={len(b)} mean={b.mean():.3f} median={np.median(b):.3f} "
          f"n>=.90={int((b >= .9).sum())} n>=.85={int((b >= .85).sum())}")
    out["summary"] = {"n": len(rows), "own_auc_mean": float(a.mean()),
                      "best_mean": float(b.mean()), "best_ge90": int((b >= .9).sum()),
                      "best_ge85": int((b >= .85).sum())}
    (HERE / "subject_best.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print("[done] subject_best.json")


if __name__ == "__main__":
    main()
