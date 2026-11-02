"""E15：决策层对比——EMA+阈值 vs 因果两态 HMM（时长先验 = 个体记忆）。

只用已到达的分数，不用未来、不用标签。三种决策规则：
  ema0   因果 EMA(h) + 0 阈值
  emac   因果 EMA(h) + 过去中位数阈值
  hmm    两态前向滤波，发射 N(mu_i, sd)（mu/sd 由训练被试 OOF 估计），
         转移概率由"平均状态时长 D"给定：p_stay = 1 - stride/D
报告逐被试 AUC / bACC / F1，并给出是否有人达到 0.90。

运行：py -3.11 test/11_individual_20261102/decide_scan.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SC = HERE / "scores"
STRIDE = 2.0


def ema(x, h):
    if h <= 1:
        return x.copy()
    a = 0.5 ** (1.0 / h)
    return lfilter([1 - a], [1, -a], x, zi=[a * x[0]])[0]


def causal_median(s, h):
    pad = np.concatenate([np.full(h - 1, np.nan), s])
    return np.nanmedian(sliding_window_view(pad, h), axis=1)


def hmm_forward2(x, mu0, mu1, sd, p_stay):
    """标准前向滤波（对数域），状态 0=relax,1=task。返回 P(task | 已到达分数)。"""
    T = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
    logT = np.log(T + 1e-12)
    la = np.log(np.array([0.5, 0.5]))
    out = np.empty(len(x))
    for t, v in enumerate(x):
        if t:
            a = la[:, None] + logT
            m = a.max()
            la = m + np.log(np.exp(a - m).sum(axis=0))
        e = -0.5 * ((v - np.array([mu0, mu1])) / sd) ** 2
        la = la + e
        m = la.max()
        la = la - (m + np.log(np.exp(la - m).sum()))
        out[t] = np.exp(la[1])
    return out


def main():
    subs = sorted(p.stem.replace("sub-", "") for p in SC.glob("sub-*.npz"))
    print(f"[load] {len(subs)} subjects: {subs}", flush=True)
    res = {}
    for q in subs:
        Z = np.load(SC / f"sub-{q}.npz")
        y = Z["y"].astype(int)
        rec = {}
        for mdl in ("lr", "lda"):
            x = Z[f"score__{mdl}"].astype(float)
            mu0, mu1, sd = float(Z[f"mu0__{mdl}"]), float(Z[f"mu1__{mdl}"]), float(Z[f"sd__{mdl}"])
            for h in (1, 2, 4, 6, 8, 12):
                s = ema(x, h)
                r0 = {"AUC": float(roc_auc_score(y, s)),
                      "bACC": float(balanced_accuracy_score(y, (s >= 0).astype(int))),
                      "F1": float(f1_score(y, (s >= 0).astype(int)))}
                rec[f"{mdl}_ema0_h{h}"] = {"AUC": r0["AUC"], "bACC": r0["bACC"], "F1": r0["F1"]}
                sc = s - causal_median(s, max(h, 10))
                rec[f"{mdl}_emac_h{h}"] = {
                    "AUC": r0["AUC"],
                    "bACC": float(balanced_accuracy_score(y, (sc >= 0).astype(int))),
                    "F1": float(f1_score(y, (sc >= 0).astype(int)))}
            for D in (20, 30, 45, 60, 90):
                p = 1 - STRIDE / D
                pr = hmm_forward2(x, mu0, mu1, sd, p)
                rec[f"{mdl}_hmm_D{D}"] = {
                    "AUC": float(roc_auc_score(y, pr)),
                    "bACC": float(balanced_accuracy_score(y, (pr >= 0.5).astype(int))),
                    "F1": float(f1_score(y, (pr >= 0.5).astype(int)))}

        # 两模型共识：按训练分布标准化后取均值，再做 HMM / EMA
        zs = []
        for mdl in ("lr", "lda"):
            zs.append((Z[f"score__{mdl}"].astype(float) - float(Z[f"mu0__{mdl}"])) /
                      float(Z[f"sd__{mdl}"]))
        c = np.mean(zs, axis=0)
        mu0c, mu1c = float(Z["mu0__lr"]) / float(Z["sd__lr"]), float(Z["mu1__lr"]) / float(Z["sd__lr"])
        for h in (1, 4, 6, 8):
            s = ema(c, h)
            rec[f"cons_ema0_h{h}"] = {
                "AUC": float(roc_auc_score(y, s)),
                "bACC": float(balanced_accuracy_score(y, (s >= 0).astype(int))),
                "F1": float(f1_score(y, (s >= 0).astype(int)))}
        for D in (20, 30, 45, 60, 90):
            pr = hmm_forward2(c, mu0c, mu1c, 1.0, 1 - STRIDE / D)
            rec[f"cons_hmm_D{D}"] = {
                "AUC": float(roc_auc_score(y, pr)),
                "bACC": float(balanced_accuracy_score(y, (pr >= 0.5).astype(int))),
                "F1": float(f1_score(y, (pr >= 0.5).astype(int)))}
        # 目标被试自己的因果标准化（过去 300 帧中位数/MAD）后再 HMM
        w = 150
        med = np.array([np.median(x[max(0, i - w):i + 1]) for i in range(len(x))])
        mad = np.array([np.median(np.abs(x[max(0, i - w):i + 1] - med[i])) + 1e-6 for i in range(len(x))])
        xn = (x - med) / (1.4826 * mad)
        for D in (30, 45, 60, 90):
            pr = hmm_forward2(xn, mu0, mu1, sd, 1 - STRIDE / D)
            rec[f"lr_cstd_hmm_D{D}"] = {
                "AUC": float(roc_auc_score(y, pr)),
                "bACC": float(balanced_accuracy_score(y, (pr >= 0.5).astype(int))),
                "F1": float(f1_score(y, (pr >= 0.5).astype(int)))}
        res[q] = rec
        best = max(rec, key=lambda k: rec[k]["bACC"])
        print(f"{q}: staticAUC={roc_auc_score(y, Z['score__lr']):.3f} "
              f"best={best} AUC={rec[best]['AUC']:.3f} bACC={rec[best]['bACC']:.3f} "
              f"F1={rec[best]['F1']:.3f}", flush=True)

    # 汇总：每个规则在 13 人上的均值与最好被试
    rules = list(next(iter(res.values())).keys())
    print("\n[rule summary] 按最好被试 bACC 排序（前 12）")
    rows = []
    for r in rules:
        b = [res[q][r]["bACC"] for q in res]
        a = [res[q][r]["AUC"] for q in res]
        best_q = max(res, key=lambda q: res[q][r]["bACC"])
        rows.append((r, float(np.mean(a)), float(np.mean(b)), best_q,
                     float(res[best_q][r]["bACC"]), float(res[best_q][r]["F1"]),
                     float(res[best_q][r]["AUC"])))
    for t in sorted(rows, key=lambda z: -z[4])[:12]:
        print(f"  {t[0]:18s} meanAUC={t[1]:.3f} meanBACC={t[2]:.3f} "
              f"best={t[3]}: bACC={t[4]:.3f} F1={t[5]:.3f} AUC={t[6]:.3f}")

    hits = [(r, q, res[q][r]["bACC"], res[q][r]["F1"], res[q][r]["AUC"])
            for r in rules for q in res if res[q][r]["bACC"] >= 0.9]
    print(f"\n[≥0.90] {len(hits)} 条")
    for h in sorted(hits, key=lambda z: -z[2])[:12]:
        print("  ", h)
    (HERE / "decide_scan.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print("[done] decide_scan.json")


if __name__ == "__main__":
    main()
