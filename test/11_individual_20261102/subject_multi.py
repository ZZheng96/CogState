"""E21：逐被试多候选 + 本人标注定符号 → 每人在"最适合的方法"下的成绩。

候选（都只用当前窗特征，决策只用过去数据）：
  own_k      本人 LR（特征级因果标准化，折内选 k∈{50,100,200,全}）
  eng        β − (α+θ)/2（engagement index，对数功率）
  beta       β 绝对（对数）功率
  tha        θ − α
  delta      δ（1–4 Hz）功率
  glb        全局 LR 分数 + 两态 HMM 记忆（E11）

符号校准：对每个候选、每个留出块，用**其余块的标注**判断该候选在本人的极性
（task 均值 > relax 均值 则 +1，否则 −1），再作用于被评块；随后 EMA(h) 平滑，阈值 0。
极性只占 1 bit，来自本人其他块的标注，属"本人标注 + 严格交叉拟合"。

输出：逐被试最优候选、AUC / bACC / F1，以及 ≥0.90 / ≥0.85 人数。

运行：py -3.11 test/11_individual_20261102/subject_multi.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from feat_cstd import running_stats
from final_eval import cstd, hmm_forward
from subject_best import fit_score, choose_k

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLID = HERE / "sliding"
SC = HERE / "scores"
STRIDE = 2.0


def ema(x, h):
    if h <= 1:
        return x.copy()
    a = 0.5 ** (1.0 / h)
    return lfilter([1 - a], [1, -a], x, zi=[a * x[0]])[0]


def main():
    man = json.loads((ROOT / "database" / "manifest.json").read_text(encoding="utf-8"))
    names = man["feature_names"] + man["emg_feature_names"]
    ix = {k: names.index(k) for k in ("stft_log_power_1_4_mean", "stft_log_power_4_8_mean",
                                      "stft_log_power_8_13_mean", "stft_log_power_13_30_mean",
                                      "stft_log_power_30_45_mean")}
    emg_cols = list(range(len(man["feature_names"]), len(names)))
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
        d_, th, al, be, ga = (Xc[:, ix[k]] for k in (
            "stft_log_power_1_4_mean", "stft_log_power_4_8_mean", "stft_log_power_8_13_mean",
            "stft_log_power_13_30_mean", "stft_log_power_30_45_mean"))
        tot = np.log(np.exp(d_) + np.exp(th) + np.exp(al) + np.exp(be) + np.exp(ga) + 1e-12)
        cands = {"eng": be - 0.5 * (al + th), "beta": be, "tha": th - al, "delta": d_,
                 "brel": be - tot, "theta": th, "alpha": al, "emg": Xc[:, emg_cols].mean(1)}

        # 每个候选：逐块留出 + 用其余块定极性与标准化
        scores = {name: np.full(len(y), np.nan) for name in cands}
        for name, v in cands.items():
            for b in np.unique(blk):
                te, tr = blk == b, blk != b
                if len(np.unique(y[tr])) < 2:
                    continue
                mu, sg = v[tr].mean(), v[tr].std() + 1e-12
                ztr = (v[tr] - mu) / sg
                m1, m0 = ztr[y[tr] == 1].mean(), ztr[y[tr] == 0].mean()
                sign = 1.0 if m1 > m0 else -1.0
                # 阈值取 0（标准化后的总体中点）：实测优于"两类均值中点"（后者被块偏移抬高）
                scores[name][te] = sign * (v[te] - mu) / sg

        # 本人模型（折内选 k）
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
        # 频带指数共识：eng/beta/tha/delta/brel 的 z 分数均值（各自已按块定极性）
        scores["bandmix"] = np.mean([scores[k] for k in ("eng", "beta", "tha", "delta", "brel")],
                                    axis=0)
        for name in scores:  # 单类折留下的 NaN 记为 0（中性），不参与判别
            scores[name] = np.where(np.isfinite(scores[name]), scores[name], 0.0)

        rec, preds = {}, {}
        for name, s in scores.items():
            for h in (1, 2, 4, 8, 12, 20, 30):
                e = ema(s, h)
                key = f"{name}_h{h}"
                preds[key] = (e >= 0).astype(int)
                rec[key] = {
                    "AUC": float(roc_auc_score(y, e)),
                    "bACC": float(balanced_accuracy_score(y, preds[key])),
                    "F1": float(f1_score(y, preds[key]))}
        Z = np.load(SC / f"sub-{q}.npz")
        sg = Z["score__lr"].astype(float)
        for Dd in (30, 60, 90):
            pr = hmm_forward(cstd(sg), float(Z["mu0__lr"]), float(Z["mu1__lr"]),
                             float(Z["sd__lr"]), 1 - STRIDE / Dd)
            rec[f"glb_D{Dd}"] = {"AUC": float(roc_auc_score(y, pr)),
                                 "bACC": float(balanced_accuracy_score(y, pr >= .5)),
                                 "F1": float(f1_score(y, pr >= .5))}
        best = max(rec, key=lambda k: rec[k]["bACC"])
        # 嵌套选择（诚实口径）：每个留出块用其余块挑候选，再看该块
        keys = list(preds)
        nested = np.empty(len(y), int)
        for b in np.unique(blk):
            te, tr = blk == b, blk != b
            kk = max(keys, key=lambda k: balanced_accuracy_score(y[tr], preds[k][tr]))
            nested[te] = preds[kk][te]
        rec["nested_bACC"] = float(balanced_accuracy_score(y, nested))
        rec["nested_F1"] = float(f1_score(y, nested))
        rec["best_key"], rec["best_bACC"], rec["best_F1"] = best, rec[best]["bACC"], rec[best]["F1"]
        out[q] = rec
        rows.append((q, rec["best_bACC"], best, rec["nested_bACC"], rec["best_F1"]))
        print(f"{q:4s} best {rec['best_bACC']:.3f} ({best}) nested {rec['nested_bACC']:.3f} "
              f"| own {rec['own_k_h2']['bACC']:.3f} beta {rec['beta_h4']['bACC']:.3f} "
              f"eng {rec['eng_h4']['bACC']:.3f} mix {rec['bandmix_h4']['bACC']:.3f} "
              f"glb {rec['glb_D60']['bACC']:.3f}", flush=True)

    b = np.array([r[1] for r in rows])
    nst = np.array([r[3] for r in rows])
    print(f"\n[per-subject best] n={len(b)} mean={b.mean():.3f} median={np.median(b):.3f} "
          f"n>=.90={int((b >= .9).sum())} n>=.85={int((b >= .85).sum())} n<.7={int((b < .7).sum())}")
    print(f"[nested select]  n={len(nst)} mean={nst.mean():.3f} median={np.median(nst):.3f} "
          f"n>=.90={int((nst >= .9).sum())} n>=.85={int((nst >= .85).sum())} n<.7={int((nst < .7).sum())}")
    worst = sorted(rows, key=lambda z: z[1])[:10]
    print("[worst 10]", [(q, round(v, 3), k) for q, v, k, _, _ in worst])
    out["summary"] = {"n": len(rows), "mean": float(b.mean()), "median": float(np.median(b)),
                      "ge90": int((b >= .9).sum()), "ge85": int((b >= .85).sum()),
                      "lt70": int((b < .7).sum()),
                      "nested_mean": float(nst.mean()), "nested_ge90": int((nst >= .9).sum()),
                      "nested_ge85": int((nst >= .85).sum())}
    (HERE / "subject_multi.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print("[done] subject_multi.json")


if __name__ == "__main__":
    main()
