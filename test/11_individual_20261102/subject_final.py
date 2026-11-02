"""E19：逐被试最终方案——特征级因果标准化 + 本人留一块模型 + 全局方向融合 + 个体记忆。

E18 发现（见 feat_cstd.json）：本人留一块 LOBO 在原始特征上 AUC 只有 0.1–0.5，
原因是块级偏移被模型当成状态方向；先对**特征**做因果标准化（过去 300 s）后，
本人 LOBO 恢复到 0.65–0.90（018 .761、036 .745、054 .889、044 .903、048 .798）。

本脚本对每个被试：
  s_own   本人 LOBO LR（cstd 特征，C=0.01）
  s_glb   全局 LR（其余 4 折被试，原始特征）
  s_mix   对 [s_glb, s_own] 做留一块融合
  各自再过 因果标准化 + 两态 HMM（D ∈ {30,60,90}），取逐人最优作为"该个体最适合的方法"。
报告固定配置（s_mix, D=60）与逐人最优两种口径的 AUC / bACC / F1。

运行：py -3.11 test/11_individual_20261102/subject_final.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.signal import lfilter

from feat_cstd import running_stats
from final_eval import cstd, hmm_forward

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLID = HERE / "sliding"
SC = HERE / "scores"
STRIDE = 2.0
DS = (30, 60, 90)


def lobo(Z, y, blk, C=0.01):
    """留一块交叉拟合；每折分数用**训练折**的均值/标准差标准化，保证跨折同尺度。"""
    s = np.full(len(y), np.nan)
    ok = 0
    for b in np.unique(blk):
        te = blk == b
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(Z[tr])
        m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Z[tr]), y[tr])
        dtr = m.decision_function(sc.transform(Z[tr]))
        s[te] = (m.decision_function(sc.transform(Z[te])) - dtr.mean()) / (dtr.std() + 1e-12)
        ok += int(te.sum())
    return s, ok


def lobo2(A, y, blk):
    """对 [s_glb, s_own] 做留一块融合。"""
    out = np.full(len(y), np.nan)
    for b in np.unique(blk):
        te = blk == b
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            out[te] = A[te, 0]
            continue
        sc = StandardScaler().fit(A[tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(A[tr]), y[tr])
        out[te] = m.decision_function(sc.transform(A[te]))
    return out


def score(y, s):
    return {"AUC": float(roc_auc_score(y, s)),
            "bACC": float(balanced_accuracy_score(y, (s >= 0).astype(int))),
            "F1": float(f1_score(y, (s >= 0).astype(int)))}


def ema(x, h):
    if h <= 1:
        return x.copy()
    a = 0.5 ** (1.0 / h)
    return lfilter([1 - a], [1, -a], x, zi=[a * x[0]])[0]


def main():
    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F = np.nan_to_num(D["F"].astype(np.float32))
    y_all, g = D["y"].astype(int), D["g"]
    folds = np.empty(len(y_all), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                .split(np.zeros(len(y_all)), y_all, g)):
        folds[te] = k
    fold_of = {q: int(folds[g == q][0]) for q in sorted(np.unique(g))}

    subs = sorted(p.stem.replace("sub-", "") for p in SC.glob("sub-*.npz"))
    rows, out = [], {}
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
        s_own, n_ok = lobo(Xc, y, blk)
        s_glb = np.load(SC / f"sub-{q}.npz")["score__lr"].astype(float)
        if n_ok < len(y):  # 单类折：回退全局分数
            s_own = np.where(np.isfinite(s_own), s_own, s_glb)
        s_mix = lobo2(np.c_[s_glb, s_own], y, blk)

        rec = {}
        for name, s in (("own", s_own), ("glb", s_glb), ("mix", s_mix)):
            rec[name] = score(y, s)
            if name == "glb":
                for Dd in DS:
                    pr = hmm_forward(cstd(s), -1.0, 1.0, 1.2, 1 - STRIDE / Dd)
                    rec[f"glb_mem{Dd}"] = {
                        "AUC": float(roc_auc_score(y, pr)),
                        "bACC": float(balanced_accuracy_score(y, pr >= .5)),
                        "F1": float(f1_score(y, pr >= .5))}
            for h in (1, 2, 4, 8):
                e = ema(s, h)
                rec[f"{name}_ema{h}"] = {
                    "AUC": float(roc_auc_score(y, e)),
                    "bACC": float(balanced_accuracy_score(y, (e >= 0).astype(int))),
                    "F1": float(f1_score(y, (e >= 0).astype(int)))}
        best = max((k for k in rec if "ema" in k or "_mem" in k),
                   key=lambda k: rec[k]["bACC"])
        rec["best_key"] = best
        rec["best_bACC"] = rec[best]["bACC"]
        rec["fixed_bACC"] = rec["own_ema4"]["bACC"]
        out[q] = rec
        rows.append((q, rec["own"]["AUC"], rec["glb"]["AUC"], rec["mix"]["AUC"],
                     rec["own_ema4"]["bACC"], rec["best_bACC"], best))
        print(f"{q:4s} own {rec['own']['AUC']:.3f} glb {rec['glb']['AUC']:.3f} "
              f"mix {rec['mix']['AUC']:.3f} | fixed(own,ema4) {rec['own_ema4']['bACC']:.3f} "
              f"| best {rec['best_bACC']:.3f} ({best})", flush=True)

    fx = np.array([r[4] for r in rows])
    bt = np.array([r[5] for r in rows])
    print(f"\n[fixed own+ema4] n={len(fx)} mean={fx.mean():.3f} median={np.median(fx):.3f} "
          f"n>=.90={int((fx >= .9).sum())} n>=.85={int((fx >= .85).sum())}")
    print(f"[per-subject best] n={len(bt)} mean={bt.mean():.3f} median={np.median(bt):.3f} "
          f"n>=.90={int((bt >= .9).sum())} n>=.85={int((bt >= .85).sum())}")
    out["summary"] = {"fixed_mean": float(fx.mean()), "fixed_ge90": int((fx >= .9).sum()),
                      "best_mean": float(bt.mean()), "best_ge90": int((bt >= .9).sum()),
                      "n": len(rows)}
    (HERE / "subject_final.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print("[done] subject_final.json")


if __name__ == "__main__":
    main()
