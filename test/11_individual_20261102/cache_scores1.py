"""E24-1：把步长 1 s 的逐被试候选分数缓存下来，供逐人微调（workup.py）反复使用。

对每个被试保存：
  y, blk, t
  score__<cand>   11 个候选（频带指数已按"其余块"定极性并标准化，块间可比）
  score__own_k    本人 LR（特征级因果标准化 + 折内选 k）
  score__bandmix  5 个频带指数的共识
  score__glb      全局 LR（其余 4 折被试）

运行：py -3.11 test/11_individual_20261102/cache_scores1.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from feat_cstd import running_stats
from subject_multi1 import SLID, choose_k, fit_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "scores1"


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

    OUT.mkdir(exist_ok=True)
    subs = sorted(p.stem.replace("sub-", "") for p in SLID.glob("sub-*.npz"))
    for q in subs:
        S = np.load(SLID / f"sub-{q}.npz")
        X = np.nan_to_num(S["F"].astype(np.float32))
        y, blk, t = S["y"].astype(int), S["blk"].astype(int), S["t"]
        o = np.argsort(t)
        X, y, blk, t = X[o], y[o], blk[o], t[o]
        if len(np.unique(y)) < 2:
            continue
        m, sd = running_stats(X, w=300)
        Xc = (X - m) / sd
        d_, th, al, be, ga = (Xc[:, ix[k]] for k in (
            "stft_log_power_1_4_mean", "stft_log_power_4_8_mean", "stft_log_power_8_13_mean",
            "stft_log_power_13_30_mean", "stft_log_power_30_45_mean"))
        tot = np.log(np.exp(d_) + np.exp(th) + np.exp(al) + np.exp(be) + np.exp(ga) + 1e-12)
        fc = np.log(np.array([2.5, 6.0, 10.5, 21.5, 37.5]))
        P = np.vstack([d_, th, al, be, ga]).T
        fc0 = fc - fc.mean()
        slope = (P - P.mean(1, keepdims=True)) @ fc0 / np.dot(fc0, fc0)
        w = np.exp(P - P.max(1, keepdims=True))
        cent = (w * np.array([2.5, 6.0, 10.5, 21.5, 37.5])).sum(1) / w.sum(1)
        cands = {"eng": be - 0.5 * (al + th), "beta": be, "tha": th - al, "delta": d_,
                 "brel": be - tot, "theta": th, "alpha": al, "emg": Xc[:, emg_cols].mean(1),
                 "slope": slope, "cent": cent, "bt": be - th}

        out = {}
        for name, v in cands.items():
            s = np.full(len(y), np.nan)
            for b in np.unique(blk):
                te, tr = blk == b, blk != b
                if len(np.unique(y[tr])) < 2:
                    continue
                mu, sg = v[tr].mean(), v[tr].std() + 1e-12
                z = (v[tr] - mu) / sg
                sign = 1.0 if z[y[tr] == 1].mean() > z[y[tr] == 0].mean() else -1.0
                s[te] = sign * (v[te] - mu) / sg
            out[f"score__{name}"] = np.nan_to_num(s, nan=0.0).astype(np.float32)
        s_own = np.full(len(y), np.nan)
        for b in np.unique(blk):
            te, tr = blk == b, blk != b
            if len(np.unique(y[tr])) < 2:
                s_own[te] = 0.0
                continue
            k = choose_k(Xc[tr], y[tr], blk[tr])
            dtr, dte = fit_score(Xc[tr], y[tr], Xc[te], k)
            s_own[te] = (dte - dtr.mean()) / (dtr.std() + 1e-12)
        out["score__own_k"] = np.nan_to_num(s_own, nan=0.0).astype(np.float32)
        out["score__bandmix"] = np.mean([out[f"score__{k}"] for k in
                                         ("eng", "beta", "tha", "delta", "brel")], axis=0).astype(np.float32)
        scg, mg = gmodels[fold_of[q]]
        out["score__glb"] = mg.decision_function(scg.transform(X)).astype(np.float32)
        np.savez_compressed(OUT / f"sub-{q}.npz", y=y.astype(np.int8), blk=blk.astype(np.int16),
                            t=t, **out)
        print(f"[{q}] saved {len(y)} frames, {len(out)} scores", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
