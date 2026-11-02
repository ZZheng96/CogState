"""E13：滑窗 + 个体记忆（因果）→ 逐被试 bACC / AUC，目标：至少 1 人 bACC ≥ 0.90。

协议：
  - 训练：其余被试的 6 s 窗（realorder.npz），标准化只在训练集拟合，被试完全隔离；
  - 推理：目标被试的 6 s 窗、2 s 步长滑窗（build_sliding.py），按真实时间排序；
  - 个体记忆：对分数做因果 EMA（半衰期 h 帧 = 2h 秒），阈值用"过去分数的中位数"
    （只用已到达数据，不用未来、不用标签）；
  - 指标：逐被试 AUC / bACC / F1；bACC 用因果阈值。

运行：py -3.11 test/11_individual_20261102/stream_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLIDING = HERE / "sliding"


def ema(x, h):
    if h <= 1:
        return x.copy()
    a = 0.5 ** (1.0 / h)
    return lfilter([1 - a], [1, -a], x, zi=[a * x[0]])[0]


def causal_thr(s, h):
    pad = np.concatenate([np.full(h - 1, np.nan), s])
    return np.nanmedian(sliding_window_view(pad, h), axis=1)


def main():
    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F = np.nan_to_num(D["F"].astype(np.float32))
    y, g = D["y"].astype(int), D["g"]
    subs = sorted(np.unique(g))
    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                .split(np.zeros(len(y)), y, g)):
        folds[te] = k
    fold_of = {q: int(folds[g == q][0]) for q in subs}

    hs = [1, 2, 4, 6, 8, 12, 16, 20, 30]
    res = {h: {} for h in hs}
    print(f"{'sub':4s} " + " ".join(f"h{h:<3d}" for h in hs))
    for q in subs:
        p = SLIDING / f"sub-{q}.npz"
        if not p.exists():
            continue
        S = np.load(p)
        Fs, ys, ts = np.nan_to_num(S["F"].astype(np.float32)), S["y"].astype(int), S["t"]
        o = np.argsort(ts)
        Fs, ys = Fs[o], ys[o]
        tr = np.flatnonzero(folds != fold_of[q])
        sc = StandardScaler().fit(F[tr])
        m = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(F[tr]), y[tr])
        raw = m.decision_function(sc.transform(Fs))
        line = f"{q:4s} "
        for h in hs:
            s = ema(raw, h)
            p0 = (s >= 0).astype(int)
            pc = (s - causal_thr(s, max(h, 10)) >= 0).astype(int)
            rec = {"AUC": float(roc_auc_score(ys, s)),
                   "bACC0": float(balanced_accuracy_score(ys, p0)),
                   "bACCc": float(balanced_accuracy_score(ys, pc)),
                   "F1_0": float(f1_score(ys, p0))}
            res[h][q] = rec
            line += f"{rec['bACC0']:.3f} "
        print(line, flush=True)

    out = {"stride_s": 2, "window_s": 6, "memory_frames": hs,
           "protocol": "训练=其余被试 6 s 窗；推理=目标被试 2 s 步长滑窗；阈值=过去分数中位数",
           "per_h": {str(h): v for h, v in res.items()}}
    print("\n" + "=" * 80)
    print(f"{'h':>3s} {'frames_s':>9s} {'poolAUC':>8s} {'meanB0':>7s} {'meanBc':>7s} "
          f"{'n>=.9(0)':>9s} {'n>=.9(c)':>9s}  best0")
    summary = {}
    for h in hs:
        v = res[h]
        if not v:
            continue
        aucs = [r["AUC"] for r in v.values()]
        b0 = [r["bACC0"] for r in v.values()]
        bc = [r["bACCc"] for r in v.values()]
        best = max(v, key=lambda k: v[k]["bACC0"])
        summary[str(h)] = {"pooled_auc": float(np.mean(aucs)),
                           "mean_bacc_zero": float(np.mean(b0)),
                           "mean_bacc_cmed": float(np.mean(bc)),
                           "n_ge_09_zero": int(sum(b >= 0.9 for b in b0)),
                           "n_ge_09_cmed": int(sum(b >= 0.9 for b in bc)),
                           "best_subject": best, "best_bacc0": float(v[best]["bACC0"]),
                           "best_baccc": float(v[best]["bACCc"]),
                           "best_auc": float(v[best]["AUC"])}
        print(f"{h:3d} {2*h:9d} {np.mean(aucs):8.4f} {np.mean(b0):7.4f} {np.mean(bc):7.4f} "
              f"{summary[str(h)]['n_ge_09_zero']:9d} {summary[str(h)]['n_ge_09_cmed']:9d}  "
              f"{best}:{v[best]['bACC0']:.3f} (AUC {v[best]['AUC']:.3f})", flush=True)
    out["summary"] = summary
    (HERE / "stream_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print("[done] stream_eval.json")


if __name__ == "__main__":
    main()
