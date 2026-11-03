"""E24-4 软分组：画像空间高斯核权重 w_j=exp(-d_ij²/2h²) 对其他被试的测试 bACC 加权，
取加权平均最高的 top-m 个配置 z 平均叠加（阈值=第 1 名配置的阈值）。
(h, m) 由内层 LOSO 选（留出 i 后，对每个 j≠i 用 k∉{i,j} 的加权结果评估 j）。
h 网格 = 画像两两距离中位数 × {.25,.5,1,2,∞}；m=1..5。特征集 cal / ax3 / ax5。
输出 soft_cluster.json、soft_per_subject.csv。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clib import HERE, AX3, AX5, load, prof_matrix, agg, StackEval, ekey, cfg_rank  # noqa: E402

R = load()
EV = StackEval(R)
nq = R["nq"]
TB3 = R["ev"][:, 0]
AVC = ~np.all(np.isnan(TB3), 2)
HS = (.25, .5, 1, 2, np.inf)
MS = (1, 2, 3, 4, 5)


def plan(j, excl, Dm, h, av):
    o = [k for k in range(nq) if k != j and k not in excl]
    w = np.ones(len(o)) if np.isinf(h) else np.exp(-Dm[j, o] ** 2 / (2 * h * h))
    order, bt, _ = cfg_rank(TB3, o, w, av)
    top = [int(c) for c in order[:max(MS)]]
    return [(ekey(top[:m]), int(bt[top[0]])) for m in MS]


def run(Pz):
    Dm = np.sqrt(((Pz[:, None] - Pz[None]) ** 2).sum(2))
    med = np.median(Dm[np.triu_indices(nq, 1)])
    hs = [h * med for h in HS]
    vals, chosen = [], []
    for i in range(nq):
        P = {(j, hi): plan(j, (i,), Dm, h, AVC[i] & AVC[j]) for j in range(nq) if j != i for hi, h in enumerate(hs)}
        EV.get_many([(j, k) for (j, _), pl in P.items() for k, _ in pl])
        sc = np.zeros((len(hs), len(MS)))
        for (j, hi), pl in P.items():
            for mi, (k, t) in enumerate(pl):
                v = EV.get(j, k)[t, 0]
                sc[hi, mi] += 0 if np.isnan(v) else v
        hi, mi = np.unravel_index(np.argmax(np.round(sc, 6)), sc.shape)
        k, t = plan(i, (), Dm, hs[hi], AVC[i])[mi]
        vals.append(tuple(float(x) for x in EV.get(i, k)[t]))
        chosen.append((HS[hi], MS[mi]))
    return vals, chosen


def main():
    res, per = {}, {}
    for fs, cols in (("cal", None), ("ax3", AX3), ("ax5", AX5)):
        v, ch = run(prof_matrix(R, cols))
        nm = f"soft_{fs}"
        per[nm] = v
        res[nm] = agg(v)
        res[nm]["h_hist"] = {str(h): sum(c[0] == h for c in ch) for h in HS}
        res[nm]["m_hist"] = {str(m): sum(c[1] == m for c in ch) for m in MS}
        a = res[nm]
        print(f"{nm:12s} bACC {a['bacc']:.3f} AUC {a['auc']:.3f} F1 {a['f1']:.3f} n90 {a['n90']}"
              f" h {a['h_hist']} m {a['m_hist']}", flush=True)
    json.dump(res, open(HERE / "soft_cluster.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(HERE / "soft_per_subject.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject"] + [f"{n}:{m}" for n in per for m in ("bacc", "auc", "f1")])
        for i, q in enumerate(R["subs"]):
            w.writerow([q] + [round(x, 4) for n in per for x in per[n][i]])


if __name__ == "__main__":
    main()
