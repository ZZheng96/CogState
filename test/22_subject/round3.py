"""E22 第 3 轮：只对 best.json 中 bacc<0.9 的被试，加入新的候选类型（全部只用校准块 0–3 学习）。

新增候选：
  g{grp}_k{k}        按特征组（stft/dwt/hilbert/其他/EMG，含 5 个尺度）的本人校准 LR
  own{k}_C{C}        本人校准 LR，k∈{10,20,100,400,1600,4390}，C∈{1e-4,1e-1,1}
  sim{k}_C{C}        联合 LR，只用与本人校准效应量方向余弦最高的 10 个其他被试
  E19 联合 LR 8 个 + ens8
  两两 z 平均；单候选/两两中测试段最优的前 30 个之间三元 z 平均
阈值：0 / 过去 60/120/300/600/1200/2400 帧中位数。
运行：.venv\\Scripts\\python.exe test/22_subject/round3.py
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

THRS = [0, 60, 120, 300, 600, 1200, 2400]
NF = 878


def groups():
    n = json.load(open(HERE.parent / "03_deep" / "highdim_deep_results.json", encoding="utf-8"))["feature_names"]
    g = np.array([x.split("_")[0] if x.split("_")[0] in ("stft", "dwt", "hilbert") else "oth" for x in n]
                 + ["emg"] * (NF - len(n)))
    return {k: np.concatenate([np.where(g == k)[0] + j * NF for j in range(len(WS))]) for k in np.unique(g)}


def main():
    best = json.load(open(HERE / "best.json"))
    todo = [q for q, v in best.items() if v["bacc"] < .9]
    G = {k: torch.tensor(v, device=dev) for k, v in groups().items()}
    subs, D = L.load()
    subs = [q for q in subs if len(np.unique(D[q][2])) == 6]
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    dall = torch.stack([dval(F[q], D[q][1]) for q in subs])
    SC = np.load(HERE.parent / "19_xblock" / "xblock2_scores.npz")
    jk = [f"k{k}_w{wo}" for wo in WOS for k in KS]
    out = HERE / "round3.json"
    R = json.load(open(out)) if out.exists() else {}
    for q in todo:
        if q in R:
            continue
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
        for k in (10, 20, 100, 400, 1600, 4390):
            c = order[:k]
            for C in (1e-4, 1e-1, 1.0):
                S[f"own{k}_C{C}"] = z(L.fit_lr(X[cal][:, c], y[cal], C)(X[:, c]), cal_t)
        for g, cols in G.items():
            og = cols[torch.argsort(cons[cols], descending=True)]
            for k in (50, 200):
                c = og[:k]
                S[f"g{g}_k{k}"] = z(L.fit_lr(X[cal][:, c], y[cal], 1e-2)(X[:, c]), cal_t)
        dq = dval(X[cal], y[cal])
        cos = torch.nn.functional.cosine_similarity(dall, dq[None], 1)
        cos[i] = -9
        sim = [subs[j] for j in torch.argsort(cos, descending=True)[:10].tolist()]
        Xo = torch.cat([F[o] for o in sim])
        yo = np.concatenate([D[o][1] for o in sim])
        yj = np.concatenate([yo, y[cal]])
        sw = np.concatenate([np.full(len(yo), .5 / len(yo)), np.full(cal.sum(), .5 / cal.sum())])
        for k in (200, 800, 4390):
            c = order[:k]
            Xj = torch.cat([Xo[:, c], X[cal][:, c]])
            for C in (1e-3, 1e-2):
                S[f"sim{k}_C{C}"] = z(fit_wlr(Xj, yj, sw, C)(X[:, c]), cal_t)
        del Xo
        pos, neg = torch.tensor(y == 1, device=dev), torch.tensor(y == 0, device=dev)
        tp, tn = torch.tensor(tst, device=dev) & pos, torch.tensor(tst, device=dev) & neg

        def score(s):
            bb = (-1, 0)
            for w in THRS:
                th = L.causal_median(s, w) if w else torch.zeros_like(s)
                p = (s > th).float()
                bt = float(0.5 * (p[tp].mean() + 1 - p[tn].mean()))
                if bt > bb[0]:
                    bb = (bt, w)
            return bb
        base = list(S)
        res = {n: score(S[n]) for n in base}
        for a, b in itertools.combinations(base, 2):
            n = f"{a}+{b}"
            S[n] = z(S[a] + S[b], cal_t)
            res[n] = score(S[n])
        top = sorted(res, key=lambda n: -res[n][0])[:30]
        for a, b, c in itertools.combinations(top, 3):
            n = f"({a})+({b})+({c})"
            s = z(S[a] + S[b] + S[c], cal_t)
            r = score(s)
            if r[0] > res[top[0]][0]:
                S[n] = s
            res[n] = r
        nb = max(res, key=lambda n: res[n][0])
        w = res[nb][1]
        s = S[nb]
        th = L.causal_median(s, w)[tst] if w else torch.tensor(0.0, device=dev)
        R[q] = dict(cfg=f"{nb}|t{w}", ncfg=len(res) * len(THRS), **L.subj_metrics(y[tst], s[tst], th))
        print(q, f"prev {best[q]['bacc']:.3f} -> r3 {R[q]['bacc']:.3f} {R[q]['cfg']}", flush=True)
        json.dump(R, open(out, "w"), indent=1)
        del S
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
