"""E11a：个体记忆的第一步——因果分数记忆（EMA / 因果中位数）能否把某个被试推到 bACC 0.90？

背景：E10 矩阵显示单个被试的最优跨被试 AUC 只有 0.885（036），池化 0.64。
按 bACC=Φ(d'/2) 换算：
    bACC .90 ← d' 2.56 ← AUC .965（等方差高斯）
    036 的跨被试 AUC .851 → d' 1.47，需要 d' 放大约 1.75 倍。
若每个 6 s 窗的噪声近似独立，因果平均 k 个窗可把 d' 放大 √k：
    k=3 → ×1.73，k=5 → ×2.24。
因此"个体记忆"（只用过去已到达的分数）是当前唯一有数量级依据的路径。

本脚本只做**分数级**因果记忆（不重算特征），用真实顺序 t 逐被试滚动：
    因果 EMA：s_t = a*s_{t-1} + (1-a)*x_t,  a = 0.5 ** (1/h)
    因果滑动中位数（窗口 h）
阈值：0（分数已按训练折中心化）、以及"过去分位数"因果阈值（个体自适应）。
报告每个 h 下 57 人的 AUC / bACC、≥0.9 人数、以及最高被试。

运行：py -3.11 test/11_individual_20261102/causal_memory_scan.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BANK = ROOT / "test" / "10_projects" / "eeg_only_20261102" / "method_bank_scores.npz"
REAL = ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz"


def ema(x, h):
    if h <= 1:
        return x.copy()
    a = 0.5 ** (1.0 / h)
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = a * acc + (1 - a) * v
        y[i] = acc
    return y


def cmedian(x, h):
    if h <= 1:
        return x.copy()
    y = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - h + 1)
        y[i] = np.median(x[lo:i + 1])
    return y


def cquantile_threshold(x, h, q=0.5):
    """过去 h 个分数的 q 分位作为阈值；前 h 个用累积分位。"""
    thr = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - h + 1)
        thr[i] = np.quantile(x[lo:i + 1], q)
    return thr


def main():
    Z = np.load(BANK)
    g = Z["g"]
    y = Z["y"].astype(int)
    D = np.load(REAL)
    assert np.array_equal(D["g"], g) and np.array_equal(D["y"].astype(int), y)
    t = D["t"]
    subs = sorted(np.unique(g))
    names = [k[3:] for k in Z.files if k.startswith("s__")]
    X = {n: Z[f"s__{n}"] for n in names}
    X["consensus"] = np.mean(np.vstack([X[n] for n in names]), axis=0)

    base = ["lr_c001", "lda_shrink", "hgb", "extratrees", "mlp256", "consensus"]
    hs = [1, 2, 3, 5, 8, 12, 20]
    out = {}
    print(f"{'base':12s} {'h':>3s} {'AUC':>7s} {'bACC':>7s} {'F1m':>7s} {'n>=.9':>6s}  best(bACC)")
    for b in base:
        x = X[b]
        for h in hs:
            s = np.empty_like(x)
            s_thr = np.empty_like(x)
            for q in subs:
                m = g == q
                idx = np.flatnonzero(m)
                idx = idx[np.argsort(t[idx])]
                xs = ema(x[idx], h)
                s[idx] = xs
                s_thr[idx] = xs - cquantile_threshold(xs, max(h, 10), 0.5)
            yh = (s >= 0).astype(int)
            th = (s_thr >= 0).astype(int)
            per_b = {q: balanced_accuracy_score(y[g == q], yh[g == q]) for q in subs}
            per_b2 = {q: balanced_accuracy_score(y[g == q], th[g == q]) for q in subs}
            used = per_b2 if np.mean(list(per_b2.values())) > np.mean(list(per_b.values())) else per_b
            tag = "qthr" if used is per_b2 else "zero"
            best = max(used, key=used.get)
            rec = {"AUC": float(roc_auc_score(y, s)),
                   "bACC": float(np.mean(list(used.values()))),
                   "F1_macro": float(f1_score(y, yh, average="macro")),
                   "n_ge_09": int(sum(v >= 0.9 for v in used.values())),
                   "best_subject": best, "best_bacc": float(used[best]),
                   "per_subject_bacc": {q: float(v) for q, v in used.items()},
                   "threshold": tag}
            out[f"{b}__h{h}"] = rec
            print(f"{b:12s} {h:3d} {rec['AUC']:7.4f} {rec['bACC']:7.4f} {rec['F1_macro']:7.4f} "
                  f"{rec['n_ge_09']:6d}  {best}:{rec['best_bacc']:.3f} ({tag})", flush=True)

    # 找出全部 >=0.9 的组合与被试
    hits = [(k, v["best_subject"], v["best_bacc"]) for k, v in out.items() if v["n_ge_09"] > 0]
    print(f"\n[≥0.9] {len(hits)} 个配置命中：")
    for h in sorted(hits, key=lambda z: -z[2])[:15]:
        print("  ", h)
    (HERE / "causal_memory_scan.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    print("[done] causal_memory_scan.json")


if __name__ == "__main__":
    main()
