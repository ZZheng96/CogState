"""E11b：记忆增益的真伪检验——打乱分数序列后，同一套「个体记忆」还能不能到 0.9？

E11a 用 h=12/20（72–120 s）的因果 EMA + 过去中位数阈值，把 054 推到 bACC 0.931、048 到 0.910。
但块是 6 块交替，长记忆可能只是把慢漂移对齐到块结构。必须排除"没用 EEG"的可能。

控制：
  real      原分数（真实顺序）
  shuffle   被试内随机打乱分数（标签/时间不动）→ 若记忆本身能产生 0.9，打乱后也应高
  time_only 用时间本身当分数（只看时钟）→ 该通道不依赖 EEG 的上界

运行：py -3.11 test/11_individual_20261102/memory_controls.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BANK = ROOT / "test" / "10_projects" / "eeg_only_20261102" / "method_bank_scores.npz"
REAL = ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz"


def ema(x, h):
    if h <= 1:
        return x.copy()
    a = 0.5 ** (1.0 / h)
    return lfilter([1 - a], [1, -a], x, zi=[a * x[0]])[0]


def causal_qthr(s, h):
    pad = np.concatenate([np.full(h - 1, np.nan), s])
    return np.nanmedian(sliding_window_view(pad, h), axis=1)


def pipeline(x, h):
    s = ema(x, h)
    return (s - causal_qthr(s, max(h, 10)) >= 0).astype(int)


def main():
    rng = np.random.default_rng(0)
    Z = np.load(BANK)
    g, y = Z["g"], Z["y"].astype(int)
    t = np.load(REAL)["t"]
    subs = sorted(np.unique(g))
    order = {q: np.flatnonzero(g == q)[np.argsort(t[g == q])] for q in subs}
    cons = np.mean(np.vstack([Z[f"s__{k[3:]}"] for k in Z.files if k.startswith("s__")]), axis=0)
    bases = {"consensus": cons, "mlp256": Z["s__mlp256"], "lr_c001": Z["s__lr_c001"]}
    out = {}
    print(f"{'base':10s} {'h':>3s} {'real_best':>10s} {'shuf_mean':>9s} {'shuf_p95':>9s} "
          f"{'shuf_max':>9s} {'n>=.9_real':>10s} {'shuf_hit':>9s}")
    for bn, x in bases.items():
        for h in (8, 12, 20):
            real = {q: balanced_accuracy_score(y[order[q]], pipeline(x[order[q]], h)) for q in subs}
            shuf, hits = [], []
            for _ in range(100):
                per = []
                for q in subs:
                    idx = order[q]
                    xs = x[idx].copy()
                    rng.shuffle(xs)
                    per.append(balanced_accuracy_score(y[idx], pipeline(xs, h)))
                shuf.append(float(np.mean(per)))
                hits.append(int((np.array(per) >= 0.9).sum()))
            best = max(real, key=real.get)
            rec = {"real": {q: float(v) for q, v in real.items()},
                   "best_subject": best, "best_bacc": float(real[best]),
                   "n_ge_09": int(sum(v >= 0.9 for v in real.values())),
                   "shuffle_mean_bacc": float(np.mean(shuf)),
                   "shuffle_p95_bacc": float(np.percentile(shuf, 95)),
                   "shuffle_max_bacc": float(np.max(shuf)),
                   "shuffle_hit_rate": float(np.mean(np.array(hits) > 0))}
            out[f"{bn}__h{h}"] = rec
            print(f"{bn:10s} {h:3d} {real[best]:10.3f} {np.mean(shuf):9.4f} "
                  f"{np.percentile(shuf,95):9.4f} {np.max(shuf):9.4f} "
                  f"{rec['n_ge_09']:10d} {rec['shuffle_hit_rate']:9.2f}", flush=True)

    per_time = {}
    for q in subs:
        idx = order[q]
        per_time[q] = balanced_accuracy_score(y[idx], pipeline(t[idx] - t[idx].mean(), 20))
    print(f"[time_only h20] 池化 bACC={np.mean(list(per_time.values())):.4f} "
          f"最高={max(per_time.values()):.3f} 命中={sum(v >= .9 for v in per_time.values())}")
    out["time_only"] = {q: float(v) for q, v in per_time.items()}
    (HERE / "memory_controls.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print("[done] memory_controls.json")


if __name__ == "__main__":
    main()
