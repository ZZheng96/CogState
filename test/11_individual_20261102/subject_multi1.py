"""E23：步长 1 s 的逐个体最优方法（更多帧 → 本人模型更稳）。

与 E12（`subject_multi.py`，步长 2 s）同协议，只改两点：
  - 滑窗步长 1 s（1700 帧/人，约为原来的 2 倍），特征仍是 6 s 窗；
  - 记忆长度 h 以**秒**给（1 s 步长下帧=秒）：{2,4,8,12,20,30,60}。

候选：8 个频带指数 + bandmix + 本人 LR（折内选 k∈{50,200,1000}）+ 全局 LR + HMM。
全局分数在此脚本内按折训练（其余 4 折被试），避免与步长 2 s 的缓存混用。

运行：py -3.11 test/11_individual_20261102/subject_multi1.py
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
SLID = HERE / "sliding1"
STRIDE = 1.0
KS = (50, 200, 1000)
HS = (2, 4, 8, 12, 20, 30, 60)


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
    n = min(2, len(np.unique(gtr)))
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
    man = json.loads((ROOT / "database" / "manifest.json").read_text(encoding="utf-8"))
    names = man["feature_names"] + man["emg_feature_names"]
    ix = {k: names.index(k) for k in ("stft_log_power_1_4_mean", "stft_log_power_4_8_mean",
                                      "stft_log_power_8_13_mean", "stft_log_power_13_30_mean",
                                      "stft_log_power_30_45_mean")}
    emg_cols = list(range(len(man["feature_names"]), len(names)))

    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F = np.nan_to_num(D["F"].astype(np.float32))
    y_all, g = D["y"].astype(int), D["g"]
    folds = np.empty(len(y_all), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                .split(np.zeros(len(y_all)), y_all, g)):
        folds[te] = k
    fold_of = {q: int(folds[g == q][0]) for q in sorted(np.unique(g))}
    gmodels = {}
    for f in range(5):
        tr = np.flatnonzero(folds != f)
        sc = StandardScaler().fit(F[tr])
        gmodels[f] = (sc, LogisticRegression(C=0.01, max_iter=3000)
                      .fit(sc.transform(F[tr]), y_all[tr]))

    subs = sorted(p.stem.replace("sub-", "") for p in SLID.glob("sub-*.npz"))
    out, rows = {}, []
    for q in subs:
        S = np.load(SLID / f"sub-{q}.npz")
        X = np.nan_to_num(S["F"].astype(np.float32))
        y, blk = S["y"].astype(int), S["blk"].astype(int)
        o = np.argsort(S["t"])
        X, y, blk = X[o], y[o], blk[o]
        if len(np.unique(y)) < 2:
            continue
        m, sd = running_stats(X, w=300)
        Xc = (X - m) / sd
        d_, th, al, be, ga = (Xc[:, ix[k]] for k in (
            "stft_log_power_1_4_mean", "stft_log_power_4_8_mean", "stft_log_power_8_13_mean",
            "stft_log_power_13_30_mean", "stft_log_power_30_45_mean"))
        tot = np.log(np.exp(d_) + np.exp(th) + np.exp(al) + np.exp(be) + np.exp(ga) + 1e-12)
        # 1/f 斜率（5 个频带 log 功率对 log 中心频率回归）与谱质心，均为免费派生量
        fc = np.log(np.array([2.5, 6.0, 10.5, 21.5, 37.5]))
        P = np.vstack([d_, th, al, be, ga]).T
        fc0 = fc - fc.mean()
        slope = (P - P.mean(1, keepdims=True)) @ fc0 / np.dot(fc0, fc0)
        w = np.exp(P - P.max(1, keepdims=True))
        cent = (w * np.array([2.5, 6.0, 10.5, 21.5, 37.5])).sum(1) / w.sum(1)
        cands = {"eng": be - 0.5 * (al + th), "beta": be, "tha": th - al, "delta": d_,
                 "brel": be - tot, "theta": th, "alpha": al, "emg": Xc[:, emg_cols].mean(1),
                 "slope": slope, "cent": cent, "bt": be - th}

        scores = {name: np.full(len(y), np.nan) for name in cands}
        for name, v in cands.items():
            for b in np.unique(blk):
                te, tr = blk == b, blk != b
                if len(np.unique(y[tr])) < 2:
                    continue
                mu, sg = v[tr].mean(), v[tr].std() + 1e-12
                z = (v[tr] - mu) / sg
                sign = 1.0 if z[y[tr] == 1].mean() > z[y[tr] == 0].mean() else -1.0
                scores[name][te] = sign * (v[te] - mu) / sg

        s_own = np.full(len(y), np.nan)
        for b in np.unique(blk):
            te, tr = blk == b, blk != b
            if len(np.unique(y[tr])) < 2:
                s_own[te] = 0.0
                continue
            k = choose_k(Xc[tr], y[tr], blk[tr])
            dtr, dte = fit_score(Xc[tr], y[tr], Xc[te], k)
            s_own[te] = (dte - dtr.mean()) / (dtr.std() + 1e-12)
        scores["own_k"] = s_own
        scores["bandmix"] = np.mean([scores[k] for k in ("eng", "beta", "tha", "delta", "brel")],
                                    axis=0)
        for name in scores:
            scores[name] = np.where(np.isfinite(scores[name]), scores[name], 0.0)

        scg, mg = gmodels[fold_of[q]]
        sg = mg.decision_function(scg.transform(X))
        rec, preds = {}, {}
        for name, s in scores.items():
            for h in HS:
                e = ema(s, h)
                key = f"{name}_h{h}"
                preds[key] = (e >= 0).astype(int)
                rec[key] = {"AUC": float(roc_auc_score(y, e)),
                            "bACC": float(balanced_accuracy_score(y, preds[key])),
                            "F1": float(f1_score(y, preds[key]))}
        for Dd in (30, 60, 90):
            # 发射参数取训练被试 OOF 的经验值（score_cache 的汇总），不使用本被试标签
            pr = hmm_forward(cstd(sg), -0.22, 0.09, 1.28, 1 - STRIDE / Dd)
            rec[f"glb_D{Dd}"] = {"AUC": float(roc_auc_score(y, pr)),
                                 "bACC": float(balanced_accuracy_score(y, pr >= .5)),
                                 "F1": float(f1_score(y, pr >= .5))}
        best = max(rec, key=lambda k: rec[k]["bACC"])
        nested = np.empty(len(y), int)
        for b in np.unique(blk):
            te, tr = blk == b, blk != b
            kk = max(preds, key=lambda k: balanced_accuracy_score(y[tr], preds[k][tr]))
            nested[te] = preds[kk][te]
        rec["nested_bACC"] = float(balanced_accuracy_score(y, nested))
        rec["best_key"], rec["best_bACC"], rec["best_F1"] = best, rec[best]["bACC"], rec[best]["F1"]
        out[q] = rec
        rows.append((q, rec["best_bACC"], best, rec["nested_bACC"]))
        print(f"{q:4s} best {rec['best_bACC']:.3f} ({best}) nested {rec['nested_bACC']:.3f}", flush=True)

    b = np.array([r[1] for r in rows])
    nst = np.array([r[3] for r in rows])
    print(f"\n[per-subject best] n={len(b)} mean={b.mean():.3f} median={np.median(b):.3f} "
          f"n>=.90={int((b >= .9).sum())} n>=.85={int((b >= .85).sum())}")
    print(f"[nested]           mean={nst.mean():.3f} median={np.median(nst):.3f} "
          f"n>=.90={int((nst >= .9).sum())} n>=.85={int((nst >= .85).sum())}")
    out["summary"] = {"n": len(rows), "mean": float(b.mean()), "median": float(np.median(b)),
                      "ge90": int((b >= .9).sum()), "ge85": int((b >= .85).sum()),
                      "nested_mean": float(nst.mean()), "nested_ge90": int((nst >= .9).sum())}
    (HERE / "subject_multi1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print("[done] subject_multi1.json")


if __name__ == "__main__":
    main()
