"""E22：逐个被试推进到 0.9（E13 思路的合规版）。

与 E13 的区别（按理论框架第 13 节）：
  - 学习阶段只用本人块 0–3（带标签）+ 其他被试数据；块 4–5 为推理测试段。
  - 推理每帧只读当前 6 s 窗特征；去掉分数 EMA（跨窗平滑不合规）。
    允许的无标签因果自适应：特征 running 标准化（5 尺度，已在特征里）、阈值 = 过去 w 帧分数中位数。
  - 极性、标准化、特征选择全部来自校准块。
候选（每人独立）：
  joint8     E19 的 8 个联合 LR（复用 xblock2_scores.npz）及其集成 ens8
  own_k_C    本人校准块一致性前 k 维 LR
  idx_m      一致性前 m 维特征（按校准块效应量符号定极性）z 分数平均 —— 对应 E13 的频带指数
  pair       上述任意两个候选的 z 分数平均
阈值：0 / 过去 300 帧中位数 / 过去 600 帧中位数。
输出两种口径：
  oracle   每人在测试段取最优配置（E13 的"每人最适合的方法"口径，含选择偏差，报告配置数）
  calsel   每人只按校准块 bACC（块 0–3 拟合内）选配置 —— 不看测试段
运行：.venv\\Scripts\\python.exe test/22_subject/subject.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "19_xblock"))
from xblock import L, WS, dev, dval, z  # noqa: E402
from xblock2 import KS, WOS  # noqa: E402

THRS = [0, 300, 600]


def bacc(y, p):
    return 0.5 * (p[y == 1].mean() + 1 - p[y == 0].mean())


def main():
    subs, D = L.load()
    subs = [q for q in subs if len(np.unique(D[q][2])) == 6]
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    SC = np.load(HERE.parent / "19_xblock" / "xblock2_scores.npz")
    jk = [f"k{k}_w{wo}" for wo in WOS for k in KS]
    R = {}
    for q in subs:
        y, blk = D[q][1], D[q][2]
        cal, tst = blk <= 3, blk >= 4
        cal_t = torch.tensor(cal, device=dev)
        X = F[q]
        p1, p2 = blk <= 1, (blk >= 2) & (blk <= 3)
        d1, d2 = dval(X[p1], y[p1]), dval(X[p2], y[p2])
        same = torch.sign(d1) == torch.sign(d2)
        cons = torch.where(same, torch.minimum(d1.abs(), d2.abs()), torch.zeros_like(d1))
        order = torch.argsort(cons, descending=True)
        S = {k: torch.tensor(SC[f"{q}|{k}"], device=dev) for k in jk}
        S["ens8"] = z(torch.stack([S[k] for k in jk]).mean(0), cal_t)
        for k in (50, 200, 800):
            c = order[:k]
            for C in (1e-3, 1e-2):
                S[f"own{k}_C{C}"] = z(L.fit_lr(X[cal][:, c], y[cal], C)(X[:, c]), cal_t)
        sg = torch.sign(d1 + d2)
        for m in (1, 3, 10, 30, 100):
            c = order[:m]
            Zc = (X[:, c] - X[cal][:, c].mean(0)) / (X[cal][:, c].std(0) + 1e-6)
            S[f"idx{m}"] = z((Zc * sg[c]).mean(1), cal_t)
        base = list(S)
        for a, b in itertools.combinations(base, 2):
            S[f"{a}+{b}"] = z(S[a] + S[b], cal_t)
        names = list(S)
        M = torch.stack([S[n] for n in names])  # (ncand, T)
        yt = torch.tensor(y, device=dev)
        res = {}
        for w in THRS:
            Th = torch.stack([L.causal_median(M[i], w) for i in range(len(names))]) if w else torch.zeros_like(M)
            P = (M > Th).float()
            pos, neg = yt == 1, yt == 0
            ba_t = 0.5 * (P[:, tst & pos.cpu().numpy()].mean(1) + 1 - P[:, tst & neg.cpu().numpy()].mean(1))
            ba_c = 0.5 * (P[:, cal & pos.cpu().numpy()].mean(1) + 1 - P[:, cal & neg.cpu().numpy()].mean(1))
            for i, n in enumerate(names):
                res[f"{n}|t{w}"] = (float(ba_t[i]), float(ba_c[i]), i, w)
        keys = list(res)
        bo = max(keys, key=lambda k: res[k][0])
        bc = max(keys, key=lambda k: res[k][1])

        def full(k):
            _, _, i, w = res[k]
            th = L.causal_median(M[i], w) if w else torch.tensor(0.0, device=dev)
            th = th[tst] if w else th
            return L.subj_metrics(y[tst], M[i][tst], th)
        R[q] = dict(oracle=dict(cfg=bo, **full(bo)), calsel=dict(cfg=bc, **full(bc)),
                    ens8=full("ens8|t0"), ncfg=len(keys))
        print(q, f"oracle {bo} {R[q]['oracle']['bacc']:.3f} | calsel {bc} {R[q]['calsel']['bacc']:.3f} "
                 f"| ens8 {R[q]['ens8']['bacc']:.3f}", flush=True)
    tab = {}
    for n in ("oracle", "calsel", "ens8"):
        tab[n] = {m: float(np.mean([R[q][n][m] for q in subs])) for m in ("bacc", "auc", "f1")}
        tab[n]["n90"] = int(sum(R[q][n]["bacc"] >= .9 for q in subs))
        print(f"  {n:7s} bACC {tab[n]['bacc']:.3f} AUC {tab[n]['auc']:.3f} F1 {tab[n]['f1']:.3f} n90 {tab[n]['n90']}")
    json.dump(dict(table=tab, per_subject=R), open(HERE / "subject.json", "w"), indent=1)


if __name__ == "__main__":
    main()
