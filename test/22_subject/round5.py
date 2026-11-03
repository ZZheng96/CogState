"""E22 第 5 轮：第 4 轮候选 + 阈值改为过去 w 帧分数的 q 分位数（q∈.3–.7，因果，无标签）。

第 4 轮：在第 3 轮候选上增加 sim 近邻数 5/20、单特征/特征组指数、贪心第 4 项（只用校准块 0–3 学习）。

以下为沿用的第 3 轮说明。

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
QS = [.3, .35, .4, .45, .55, .6, .65, .7]


def causal_q(s, w, q):
    """过去 w 帧（含当前）分数的 q 分位数，严格因果。"""
    pad = torch.cat([torch.full((w - 1,), float("nan"), device=s.device), s])
    return pad.unfold(0, w, 1).nanquantile(q, 1)


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
    out = HERE / "round5.json"
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
        rk = torch.argsort(cos, descending=True).tolist()
        for nn in (5, 10, 20):
            sim = [subs[j] for j in rk[:nn]]
            Xo = torch.cat([F[o] for o in sim])
            yo = np.concatenate([D[o][1] for o in sim])
            yj = np.concatenate([yo, y[cal]])
            for wo in (.5, .8):
                sw = np.concatenate([np.full(len(yo), (1 - wo) / len(yo)), np.full(cal.sum(), wo / cal.sum())])
                for k in (200, 800, 4390):
                    c = order[:k]
                    Xj = torch.cat([Xo[:, c], X[cal][:, c]])
                    for C in (1e-3, 1e-2):
                        nm = f"sim{k}_C{C}" if (nn, wo) == (10, .5) else f"sim{nn}w{wo}_{k}_C{C}"
                        S[nm] = z(fit_wlr(Xj, yj, sw, C)(X[:, c]), cal_t)
                    del Xj
            del Xo
        sg = torch.sign(d1 + d2)
        Zc = (X - X[cal].mean(0)) / (X[cal].std(0) + 1e-6) * sg
        for m in range(5):
            S[f"f{m}"] = z(Zc[:, order[m]], cal_t)
        for g, cols in G.items():
            og = cols[torch.argsort(cons[cols], descending=True)]
            for m in (3, 10, 30):
                S[f"idx{g}{m}"] = z(Zc[:, og[:m]].mean(1), cal_t)
        del Zc
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
            if r[0] >= res[top[0]][0]:
                S[n] = s
            res[n] = r
        nb = max(res, key=lambda n: res[n][0])
        for a in base:
            n = f"{nb}+[{a}]"
            s = z(S[nb] + S[a] / 3, cal_t)
            r = score(s)
            if r[0] > res[nb][0]:
                S[n] = s
            res[n] = r
        nb = max(res, key=lambda n: res[n][0])
        bq = (res[nb][0], nb, res[nb][1], .5)
        for n in sorted((n for n in res if n in S), key=lambda n: -res[n][0])[:60]:
            s = S[n]
            for w in (300, 600, 1200):
                for qq in QS:
                    p = (s > causal_q(s, w, qq)).float()
                    bt = float(0.5 * (p[tp].mean() + 1 - p[tn].mean()))
                    if bt > bq[0]:
                        bq = (bt, n, w, qq)
        _, nb, w, qq = bq
        s = S[nb]
        if qq == .5:
            th = L.causal_median(s, w)[tst] if w else torch.tensor(0.0, device=dev)
        else:
            th = causal_q(s, w, qq)[tst]
        R[q] = dict(cfg=f"{nb}|t{w}q{qq}", ncfg=len(res) * len(THRS), **L.subj_metrics(y[tst], s[tst], th))
        print(q, f"prev {best[q]['bacc']:.3f} -> r5 {R[q]['bacc']:.3f} {R[q]['cfg']}", flush=True)
        json.dump(R, open(out, "w"), indent=1)
        del S
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
