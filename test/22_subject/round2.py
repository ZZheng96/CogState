"""E22 第 2 轮：只对第 1 轮 oracle<0.9 的被试扩大候选族（全部只用校准块学习）。

新增候选：
  jw{wo}_k{k}_C{C}   联合 LR，wo∈{.3,.95}，k∈{200,800,1600}，C∈{1e-3,1e-1}
  sc{w}_k{k}         只用单个 running 标准化尺度 w 的特征，本人校准 LR（一致性前 k 维）
  第 1 轮的基础候选（joint8/ens8/own/idx）
  以上任意两两 z 平均
阈值：0 / 过去 120/300/600/1200 帧中位数。
运行：.venv\\Scripts\\python.exe test/22_subject/round2.py
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
from xblock import L, WS, dev, dval, z, fit_wlr  # noqa: E402
from xblock2 import KS, WOS  # noqa: E402

THRS = [0, 120, 300, 600, 1200]
NF = 878


def main():
    R1 = json.load(open(HERE / "subject.json"))["per_subject"]
    todo = [q for q, v in R1.items() if v["oracle"]["bacc"] < .9]
    subs, D = L.load()
    subs = [q for q in subs if len(np.unique(D[q][2])) == 6]
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    Xall = torch.cat([F[q] for q in subs])
    yall = np.concatenate([D[q][1] for q in subs])
    sid = np.concatenate([[i] * len(D[q][1]) for i, q in enumerate(subs)])
    SC = np.load(HERE.parent / "19_xblock" / "xblock2_scores.npz")
    jk = [f"k{k}_w{wo}" for wo in WOS for k in KS]
    R = {}
    for q in todo:
        i = subs.index(q)
        y, blk = D[q][1], D[q][2]
        cal, tst = blk <= 3, blk >= 4
        cal_t = torch.tensor(cal, device=dev)
        X = F[q]
        p1, p2 = blk <= 1, (blk >= 2) & (blk <= 3)
        d1, d2 = dval(X[p1], y[p1]), dval(X[p2], y[p2])
        cons = torch.where(torch.sign(d1) == torch.sign(d2), torch.minimum(d1.abs(), d2.abs()), torch.zeros_like(d1))
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
        for j, w in enumerate(WS):
            oj = torch.argsort(cons[j * NF:(j + 1) * NF], descending=True) + j * NF
            for k in (50, 200):
                c = oj[:k]
                S[f"sc{w}_k{k}"] = z(L.fit_lr(X[cal][:, c], y[cal], 1e-2)(X[:, c]), cal_t)
        oth = torch.tensor(sid != i, device=dev)
        Xo, yo = Xall[oth], yall[sid != i]
        yj = np.concatenate([yo, y[cal]])
        for wo in (0.3, 0.95):
            sw = np.concatenate([np.full(len(yo), (1 - wo) / len(yo)), np.full(cal.sum(), wo / cal.sum())])
            for k in (200, 800, 1600):
                c = order[:k]
                Xj = torch.cat([Xo[:, c], X[cal][:, c]])
                for C in (1e-3, 1e-1):
                    S[f"jw{wo}_k{k}_C{C}"] = z(fit_wlr(Xj, yj, sw, C)(X[:, c]), cal_t)
                del Xj
        base = list(S)
        for a, b in itertools.combinations(base, 2):
            S[f"{a}+{b}"] = z(S[a] + S[b], cal_t)
        names = list(S)
        yt = y
        pos, neg = yt == 1, yt == 0
        best = (-1, None)
        bestc = (-1, None)
        for n in names:
            s = S[n]
            for w in THRS:
                th = L.causal_median(s, w) if w else torch.zeros_like(s)
                p = (s > th).float().cpu().numpy()
                bt = 0.5 * (p[tst & pos].mean() + 1 - p[tst & neg].mean())
                bc = 0.5 * (p[cal & pos].mean() + 1 - p[cal & neg].mean())
                if bt > best[0]:
                    best = (bt, (n, w))
                if bc > bestc[0]:
                    bestc = (bc, (n, w))

        def full(nw):
            n, w = nw
            s = S[n]
            th = L.causal_median(s, w)[tst] if w else torch.tensor(0.0, device=dev)
            return dict(cfg=f"{n}|t{w}", **L.subj_metrics(y[tst], s[tst], th))
        R[q] = dict(oracle=full(best[1]), calsel=full(bestc[1]), ncfg=len(names) * len(THRS))
        print(q, f"r1 {R1[q]['oracle']['bacc']:.3f} -> r2 {R[q]['oracle']['bacc']:.3f} {R[q]['oracle']['cfg']}",
              flush=True)
        json.dump(R, open(HERE / "round2.json", "w"), indent=1)
        del S
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
