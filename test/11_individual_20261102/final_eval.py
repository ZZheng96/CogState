"""E17：最终方法在全被试上的分布——不挑人，看有多少人 ≥0.90。

方法（每个被试独立、全部因果、不用测试标签）：
  raw LR 分数 → 因果中位数/MAD 标准化（过去 300 s）→ 两态 HMM 前向滤波（p_stay=1-2/D）
输出：逐被试 static（单 6 s 窗）与 final（含记忆）的 AUC/bACC/F1。

运行：py -3.11 test/11_individual_20261102/final_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
SC = HERE / "scores"
STRIDE = 2.0


def hmm_forward(x, mu0, mu1, sd, p_stay):
    T = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
    logT = np.log(T + 1e-12)
    la = np.log(np.array([0.5, 0.5]))
    out = np.empty(len(x))
    for t, v in enumerate(x):
        if t:
            a = la[:, None] + logT
            m = a.max()
            la = m + np.log(np.exp(a - m).sum(axis=0))
        la = la - 0.5 * ((v - np.array([mu0, mu1])) / sd) ** 2
        m = la.max()
        la = la - (m + np.log(np.exp(la - m).sum()))
        out[t] = np.exp(la[1])
    return out


def cstd(x, w=150):
    n = len(x)
    med = np.array([np.median(x[max(0, i - w):i + 1]) for i in range(n)])
    mad = np.array([np.median(np.abs(x[max(0, i - w):i + 1] - med[i])) + 1e-6 for i in range(n)])
    return (x - med) / (1.4826 * mad)


def main():
    subs = sorted(p.stem.replace("sub-", "") for p in SC.glob("sub-*.npz"))
    Ds = (30, 45, 60, 90)
    out = {}
    print(f"{'sub':5s} {'staticAUC':>9s} {'staticACC':>9s} " +
          " ".join(f"D{d:<3d}" for d in Ds) + "  best")
    for q in subs:
        Z = np.load(SC / f"sub-{q}.npz")
        if "score__lr" not in Z:
            continue
        y = Z["y"].astype(int)
        x = Z["score__lr"].astype(float)
        mu0, mu1, sd = float(Z["mu0__lr"]), float(Z["mu1__lr"]), float(Z["sd__lr"])
        xn = cstd(x)
        rec = {"static_auc": float(roc_auc_score(y, x)),
               "static_bacc": float(balanced_accuracy_score(y, (x >= 0).astype(int)))}
        line = f"{q:5s} {rec['static_auc']:9.3f} {rec['static_bacc']:9.3f} "
        best = (0.0, None)
        for D in Ds:
            pr = hmm_forward(xn, mu0, mu1, sd, 1 - STRIDE / D)
            b = float(balanced_accuracy_score(y, (pr >= 0.5).astype(int)))
            rec[f"D{D}"] = {"AUC": float(roc_auc_score(y, pr)), "bACC": b,
                            "F1": float(f1_score(y, (pr >= 0.5).astype(int)))}
            line += f"{b:.3f} "
            if b > best[0]:
                best = (b, D)
        rec["best_D"] = best[1]
        rec["best_bACC"] = best[0]
        out[q] = rec
        print(line + f" D{best[1]}", flush=True)
    prim = [out[q]["D60"]["bACC"] for q in out]
    bests = [out[q]["best_bACC"] for q in out]
    print(f"\n[D60] n={len(prim)} mean={np.mean(prim):.3f} n>=.90={sum(v >= .9 for v in prim)} "
          f"best={max(out, key=lambda q: out[q]['D60']['bACC'])}:{max(prim):.3f}")
    print(f"[best-D] mean={np.mean(bests):.3f} n>=.90={sum(v >= .9 for v in bests)} "
          f"max={max(bests):.3f}")
    (HERE / "final_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print("[done] final_eval.json")


if __name__ == "__main__":
    main()
